import logging
import math
import os
import platform
import random
import threading
import time
from multiprocessing import Array
from queue import Empty

import SimpleITK as sitk
import numpy as np
import skimage.measure as measure
import torch.utils.data
from skimage.transform import resize
from torch.multiprocessing import Queue, Value

from utils.model_utils import binary_mask2bbox
from utils.project_utils import read_dict_csv, timeout


class TaskListQueue:
    """
    This is a multi-threaded version(I/O bound task) of TaskList producer.
    """

    def __init__(self, config, subset, logger: logging.Logger, max_num_files=0, shuffle_files=True):
        if subset not in ['train', 'eval', 'test']:
            logger.critical('subset should be among train, eval and test')
        if config.get('data') is None:
            logger.critical('config file does not contain data config')
            exit(1)
        self.subset = subset
        os_name = platform.system()
        data_root = config['data']['data_root'][os_name.lower()]
        instance_filename = os.path.join(data_root, config['data']['instance_list_file'])
        self.data_root = data_root
        if not os.path.exists(instance_filename):
            logger.critical('data_list_file does not exist: %s' % instance_filename)
        instances = read_dict_csv(instance_filename)
        instances = [item for item in instances if item['subset'] == subset]
        if max_num_files != 0:
            instances = instances[:max_num_files]
        if len(instances) == 0:
            logger.critical('cannot read instances from %s' % instance_filename)

        self.task_queue = Queue()
        if shuffle_files:
            random.shuffle(instances)
        self.shuffle_files = shuffle_files
        for item in instances:
            self.task_queue.put(item, timeout=30)
        self.instances = instances

        if config['data'].get('num_io_workers') is None:
            self.num_workers = 1
        elif config['data'].get('num_io_workers') <= 0:
            logger.error('you need to set num_io_worker to a positive number. The program will use 1 worker.')
            self.num_workers = 1
        else:
            self.num_workers = config['data'].get('num_io_workers')

        self.data_queue = Queue(maxsize=config['data']['num_proc_workers'])
        self.has_run_before = Value('i', 0)
        self.thread_workers = []
        for i in range(self.num_workers):
            self.thread_workers.append(
                threading.Thread(target=self._produce, name=('%sTaskProducer%d' % (subset, i)).title(), args=(i,)))
        self.thread_states = Array('i', [0] * self.num_workers)
        logger.info('The %s TaskListProducer will read %d files with %d workers for %s'
                    % (subset, len(instances), self.num_workers, subset))
        self.logger = logger
        self.reading_file_fields = {**config['data']['features'], **config['data']['labels']}
        self.spacing = config['data'].get('img_spacing')
        if self.spacing is None:
            logger.debug('keep original image spacing and do not normalize the size')

    @property
    def is_feeding(self):
        for state in self.thread_states:
            if state == 1:
                return True
        return False

    @property
    def is_all_consumed(self):
        if self.has_run_before.value == 1 and not self.is_feeding and self.data_queue.empty():
            return True
        return False

    def start(self):
        if self.has_run_before.value == 1:
            return
        self.has_run_before.value = 1
        self.logger.info('The %s TaskListProducer begins to read files in one epoch...' % self.subset)
        for i in range(self.num_workers):
            self.thread_states[i] = 1
            self.thread_workers[i].start()
        time.sleep(5)  # wait enough time for all workers to start.

    def stop_and_clear(self):
        if self.is_feeding:
            self.logger.critical('The %s TaskListProducer is still working and cannot be stopped' % self.subset)
            exit(1)
        self.thread_workers = []
        for i in range(self.num_workers):
            self.thread_workers.append(threading.Thread(target=self._produce, name='TaskProducer%d' % i, args=(i,)))
        self.logger.debug('reset the %s TaskListProducer with %d workers' % (self.subset, self.num_workers))
        self.has_run_before.value = 0
        self.task_queue.close()
        self.data_queue.close()
        self.task_queue = Queue()
        if self.shuffle_files:
            random.shuffle(self.instances)
        for ins in self.instances:
            self.task_queue.put(ins, timeout=30)
        self.data_queue = Queue(maxsize=self.num_workers + 1)

    def get_data_queue(self):
        return self.data_queue

    def _produce(self, thread_id):
        reader = sitk.ImageFileReader()
        max_retry_num = 5

        @timeout(30)
        def _read_file(_reader, _filename):
            _reader.SetFileName(_filename)
            _itk_image = _reader.Execute()
            return _itk_image

        while True:
            try:
                ins = self.task_queue.get_nowait()
            except Empty:
                break
            data = {}
            ins_loading_error = False

            for file_filed, img_type in self.reading_file_fields.items():
                if ins_loading_error:
                    break
                img = dict()
                retry_count = 0
                is_loaded = False
                while retry_count < max_retry_num and not is_loaded:
                    try:
                        itk_image = _read_file(reader, os.path.join(self.data_root, ins[file_filed]))
                        # not using 32bit to shrink queue size
                        if img_type == 'mask':
                            dtype = np.uint8
                        elif img_type == 'image':
                            dtype = np.int16
                        else:
                            self.logger.error('unrecognized feature or label type: %s' % img_type)
                            dtype = np.int16
                        img['data'] = sitk.GetArrayFromImage(itk_image).astype(dtype)
                        img['original_spacing'] = np.array(itk_image.GetSpacing(), np.float32)[[2, 1, 0]]
                        img['original_size'] = np.array(itk_image.GetSize(), np.int32)[[2, 1, 0]]
                        img['origin'] = itk_image.GetOrigin()
                        img['direction'] = itk_image.GetDirection()
                        img['spacing'] = self.spacing
                        # resize all imgs to normalized spacing if self.spacing is not None
                        if self.spacing is None:
                            img['spacing'] = img['original_spacing']
                            img['size'] = img['original_size']
                        else:
                            if img_type != 'mask':
                                img['data'] = resize_image(img['data'], img['original_spacing'], self.spacing)
                            else:
                                img['data'] = resize_segmentation(img['data'], img['original_spacing'], self.spacing)
                            img['size'] = np.array(img['data'].shape)
                        is_loaded = True
                    except Exception as e:
                        if retry_count < max_retry_num:
                            retry_count += 1
                            self.logger.warning('Cannot read from file %s: %s. Retry time %d'
                                                % (os.path.join(self.data_root, ins[file_filed]), e, retry_count))
                            continue
                        else:
                            self.logger.warning('Cannot read from file %s: %s. Skip this instance.'
                                                % (os.path.join(self.data_root, ins[file_filed]), e))
                            ins_loading_error = True
                            break
                if is_loaded:
                    data[file_filed] = img
                else:
                    ins_loading_error = True
            if ins_loading_error:
                continue
            else:
                ins['data'] = data
                self.data_queue.put(ins, timeout=3600 * 2)
                self.logger.debug(ins['id'] + ' just loaded')
        self.thread_states[thread_id] = 0


class AneurysmSegDataset(torch.utils.data.IterableDataset):
    """
    Iterable Dataset with multi-process prefetch image file function.
    """

    def __init__(self,
                 config,
                 subset,
                 task_list_queue: TaskListQueue,
                 logger: logging.Logger
                 ):
        super(AneurysmSegDataset).__init__()

        if config.get('data') is None:
            logger.critical('config file does not contain data config')
            exit(1)
        self.logger = logger
        self.task_list_queue = task_list_queue
        if 'img_spacing' in config['data'] and config['data']['img_spacing'] is not None:
            self.spacing = np.array(config['data']['img_spacing'], np.float32)
        else:
            self.spacing = None
        if subset not in ['train', 'eval']:
            logger.critical('Unrecognized subset: %s' % subset)
            exit(1)
        self.subset = subset
        self.config = config

    def __iter__(self):
        time.sleep(random.random())
        if self.task_list_queue.is_all_consumed:
            self.task_list_queue.stop_and_clear()
        if self.task_list_queue.has_run_before.value == 0:
            self.task_list_queue.start()

        since = time.time()
        while True:
            try:
                instance = self.task_list_queue.get_data_queue().get_nowait()
            except Empty:
                if self.task_list_queue.is_all_consumed:
                    return
                if time.time() - since > 1800:
                    self.logger.debug('Failed to get data from data_queue for 1800s and it is not all consumed')
                    return
                time.sleep(2)
                continue

            # patch samples from imgs
            if self.subset == 'train':
                pos_neg_ratio = self.config['data'].get('train_pos_neg_ratio', [1, 1])
                for i, item in enumerate(
                        ane_seg_patch_generator(instance, self.config, self.logger, sliding_window=False,
                                                balance_label=True, data_aug=True, pos_neg_ratio=pos_neg_ratio)):
                    if self.config['data'].get('debug_mode', False) and i > 5:
                        break
                    if item is None:
                        continue
                    yield item
            else:
                pos_neg_ratio = self.config['data'].get('eval_pos_neg_ratio', [1, 1])
                for i, item in enumerate(
                        ane_seg_patch_generator(instance, self.config, self.logger, sliding_window=False,
                                                balance_label=True, data_aug=False,
                                                random_seed=10, pos_neg_ratio=pos_neg_ratio)):
                    if self.config['data'].get('debug_mode', False) and i > 5:
                        break
                    if item is None:
                        continue
                    yield item


class AneurysmSegTestDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 config,
                 logger: logging.Logger
                 ):
        super(AneurysmSegTestDataset).__init__()

        if config.get('data') is None:
            logger.critical('config file does not contain data config')
            exit(1)
        self.logger = logger
        if 'img_spacing' in config['data'] and config['data']['img_spacing'] is not None:
            self.spacing = np.array(config['data']['img_spacing'], np.float32)
        else:
            self.spacing = None
        self.config = config

        self.patch_starts = None
        self.patch_size = config['data']['patch_size']
        self.overlap_step = config['data']['overlap_step']
        self.with_global = config['model']['with_global']
        self.img = None
        self.itk_image = None
        self.meta = None

    def load(self, input_file_s, input_type):
        if input_type == 'dcm':
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(input_file_s)
            ins_id = os.path.basename(os.path.dirname(input_file_s[0]))
        elif input_type == 'nii':
            reader = sitk.ImageFileReader()
            reader.SetFileName(input_file_s)
            ins_id = os.path.basename(input_file_s).split('.')[0]
        self.itk_image = reader.Execute()

        self.img = {}
        self.img['data'] = sitk.GetArrayFromImage(self.itk_image).astype(np.float32)
        self.img['original_spacing'] = np.array(self.itk_image.GetSpacing(), np.float32)[[2, 1, 0]]
        self.img['original_size'] = np.array(self.itk_image.GetSize(), np.int32)[[2, 1, 0]]
        self.img['origin'] = self.itk_image.GetOrigin()
        self.img['direction'] = self.itk_image.GetDirection()
        self.img['spacing'] = self.spacing
        if self.spacing is None:
            self.img['spacing'] = self.img['original_spacing']
            self.img['size'] = self.img['original_size']
        else:
            self.img['data'] = resize_image(self.img['data'], self.img['original_spacing'], self.spacing)
            self.img['size'] = np.array(self.img['data'].shape)

        self.meta = {'id': ins_id, 'hospital': 'unknown', 'spacing': self.img['spacing']}

        self.patch_starts = get_sliding_window_patch_starts(self.img['data'], self.patch_size, self.overlap_step)

    def __iter__(self):
        input_glo_img = self.img['data']
        brain_mask_glo_img = np.ones(self.img['size'], np.int32)
        if self.with_global:
            global_localizer = GlobalLocalizer(brain_mask_glo_img)
            cut_input_glo_img = global_localizer.cut_edge(input_glo_img, self.patch_size, is_mask=False)

        def _gen_patch(_starts):
            _ends = [_starts[i] + self.patch_size[i] for i in range(3)]
            patch_cta_img = input_glo_img[_starts[0]:_ends[0], _starts[1]:_ends[1], _starts[2]:_ends[2]].copy()
            patch_brain_mask_img = brain_mask_glo_img[_starts[0]:_ends[0], _starts[1]:_ends[1],
                                   _starts[2]:_ends[2]].copy()

            if self.with_global:
                global_cta_img = cut_input_glo_img.copy()
                global_location_bbox = global_localizer.get_position_bbox(_starts, _ends, self.patch_size)

            inputs = {'cta_img': patch_cta_img, 'brain_mask': patch_brain_mask_img}
            if self.with_global:
                inputs['global_cta_img'] = global_cta_img
                inputs['global_patch_location_bbox'] = global_location_bbox

            meta = self.meta.copy()
            meta['patch_starts'] = np.asarray(_starts)
            return inputs, meta

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.patch_starts)
        else:
            per_worker = int(math.ceil(len(self.patch_starts) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.patch_starts))
        for starts in self.patch_starts[iter_start:iter_end]:
            yield _gen_patch(starts)

    def restore_spacing(self, prediction, is_mask=True):
        if self.spacing is not None:
            if is_mask:
                return resize_segmentation(prediction, new_shape=self.img['original_size'])
            else:
                return resize_image(prediction, new_shape=self.img['original_size'])
        return prediction


class AneurysmSegTestManager:
    def __init__(self, config, logger, devices):
        self.test_dataset = AneurysmSegTestDataset(config, logger)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=config['train']['batch_size'] * len(devices),
                                                       num_workers=config['data']['num_proc_workers'], drop_last=False)

    @property
    def instance_shape(self):
        return self.test_dataset.img['size']

    @property
    def patch_starts(self):
        return self.test_dataset.patch_starts

    @property
    def patch_size(self):
        return self.test_dataset.patch_size

    def load(self, input_file_s, input_type):
        self.test_dataset.load(input_file_s, input_type)

    def save_prediction(self, prediction, output_file):
        pred_itk_image = sitk.GetImageFromArray(prediction)
        pred_itk_image.CopyInformation(self.test_dataset.itk_image)
        sitk.WriteImage(pred_itk_image, output_file)

    def restore_spacing(self, prediction, is_mask=True):
        return self.test_dataset.restore_spacing(prediction, is_mask)


def ane_seg_patch_generator(data, config, logger, sliding_window=False, balance_label=True,
                            data_aug=True, random_seed=None, pos_neg_ratio=(1, 1)):
    """
    yield patches of aneurysm segmentation
    :param data: images dict
    :param config: config dict
    :param sliding_window: false to random select negative samples
    :param balance_label: if true, repeat positive samples to balance labels.
    :param data_aug: useful for training
    :param pos_neg_ratio: only work if balance_label is true
    :return: input_patch, label_patch
    """
    data_glo = data['data']
    meta = {'id': data['id'], 'hospital': data['hospital'], 'spacing': data['data']['cta_img_file']['spacing']}
    label_glo_img = data_glo['aneurysm_seg_file']['data'].astype(np.int32)
    input_glo_img = data_glo['cta_img_file']['data'].astype(np.float32)
    if 'brain_mask_file' in data_glo:
        brain_mask_glo_img = data_glo['brain_mask_file']['data'].astype(np.int32)
    else:
        brain_mask_glo_img = np.ones(label_glo_img.shape, np.int32)
    patch_size = config['data']['patch_size']
    overlap_step = config['data']['overlap_step']
    with_global = config['model']['with_global']
    assert len(patch_size) == len(overlap_step)
    if label_glo_img.shape != input_glo_img.shape or label_glo_img.shape != brain_mask_glo_img.shape:
        logger.warning(
            'Subject %s has different shapes among cta_img, brain_mask_img and aneyrysm_seg_img' % data['id'])
        return None
    if any([label_glo_img.shape[i] < patch_size[i] for i in range(3)]):
        logger.warning('Subject %s is too small and cannot fit in one patch.' % data['id'])
        return None

    if with_global:
        global_localizer = GlobalLocalizer(brain_mask_glo_img)
        cut_input_glo_img = global_localizer.cut_edge(input_glo_img, [96, 96, 96], is_mask=False)

    def _gen_patch(_starts):
        _ends = [_starts[i] + patch_size[i] for i in range(3)]
        patch_cta_img = input_glo_img[_starts[0]:_ends[0], _starts[1]:_ends[1], _starts[2]:_ends[2]].copy()
        patch_brain_mask_img = brain_mask_glo_img[_starts[0]:_ends[0], _starts[1]:_ends[1], _starts[2]:_ends[2]].copy()
        patch_label_img = label_glo_img[_starts[0]:_ends[0], _starts[1]:_ends[1], _starts[2]:_ends[2]].copy()
        if patch_cta_img.shape != patch_brain_mask_img.shape or patch_cta_img.shape != patch_label_img.shape:
            logger.warning('Different shapes for patch_cta_img, patch_brain_mask_img and patch_label_img: %s, %s, %s'
                           % (patch_cta_img.shape, patch_brain_mask_img.shape, patch_label_img.shape))
            return None
        if with_global:
            global_cta_img = cut_input_glo_img.copy()
            global_location_mask = global_localizer.get_position_map(_starts, _ends, [96, 96, 96])
            global_label = np.array(1 if patch_label_img.sum() > 3 else 0)
        if data_aug:
            patch_cta_img += np.random.normal(0.0, 1.0, patch_cta_img.shape)
            all_arrays = [patch_cta_img, patch_brain_mask_img, patch_label_img]
            bundle = np.stack(all_arrays)
            ran_1 = [np.random.rand() > 0.5 for _ in range(3)]
            bundle = random_flip_all(bundle, ran_1)

            patch_cta_img, patch_brain_mask_img, patch_label_img = np.split(bundle, 3)
            patch_cta_img = np.squeeze(patch_cta_img.copy(), 0)
            patch_brain_mask_img = np.squeeze(patch_brain_mask_img.copy(), 0)
            patch_label_img = np.squeeze(patch_label_img.copy(), 0)

            if with_global:
                global_arrays = [global_cta_img, global_location_mask]
                global_bundle = np.stack(global_arrays)
                global_bundle = random_flip_all(global_bundle, ran_1)

                global_cta_img, global_location_mask = np.split(global_bundle, 2)
                global_cta_img = np.squeeze(global_cta_img.copy(), 0)
                global_location_mask = np.squeeze(global_location_mask.copy(), 0)

        if with_global:
            global_location_bbox = binary_mask2bbox(global_location_mask)

        inputs = {'cta_img': patch_cta_img, 'brain_mask': patch_brain_mask_img}
        targets = {'aneurysm_seg': patch_label_img}
        if with_global:
            inputs['global_cta_img'] = global_cta_img
            inputs['global_patch_location_bbox'] = global_location_bbox
            targets['global_aneurysm_label'] = global_label
        return inputs, targets, meta

    if not sliding_window:
        # compute patches number (50-300 samples per study)
        sum_brain_mask_number = 1 * np.sum(brain_mask_glo_img) // (patch_size[0] * patch_size[1] * patch_size[2])
        logger.debug('number of patches generated in %s is %s' % (data['id'], sum_brain_mask_number))

        pos_region_centers = get_positive_region_centers(label_glo_img)
        num_pos_region = len(pos_region_centers)
        count = 0
        index_pos = 0
        random_shakes = [int(patch_size[i] * 0.3) for i in range(3)]
        if random_seed is not None:
            np.random.seed(random_seed)
        while count < sum_brain_mask_number:
            # positive sample
            for _ in range(pos_neg_ratio[0]):
                if balance_label and index_pos < num_pos_region:
                    starts = [min(
                        max(0, int(pos_region_centers[index_pos][i]) - patch_size[i] // 2
                            + np.random.randint(1 - random_shakes[i], random_shakes[i]))
                        , label_glo_img.shape[i] - patch_size[i]) for i in range(3)]
                    yield _gen_patch(starts)
                    count += 1
                    index_pos = (index_pos + 1) % num_pos_region
                else:
                    index_pos += 1
            # negative sample
            for _ in range(pos_neg_ratio[1]):
                patch_found = False  # only yield samples whose centers hit the reference_mask
                while not patch_found:
                    starts = [np.random.randint(0, brain_mask_glo_img.shape[i] - patch_size[i] + 1) for i in range(3)]
                    if brain_mask_glo_img[
                        starts[0] + patch_size[0] // 2, starts[1] + patch_size[1] // 2, starts[2] + patch_size[
                            2] // 2] > 0:
                        # avoid inputing all black imgs
                        clipped_mask = (input_glo_img[starts[0]:starts[0] + patch_size[0],
                                        starts[1]:starts[1] + patch_size[1],
                                        starts[2]:starts[2] + patch_size[2]] > 0).astype(np.float32)
                        if np.mean(clipped_mask) > 0.05:
                            patch_found = True
                            yield _gen_patch(starts)
                            count += 1
                        else:
                            logger.debug('all black inputs')
    # sliding window
    else:
        assert not balance_label, 'sliding_window do not support balance label now'
        starts_list = get_sliding_window_patch_starts(input_glo_img, patch_size, overlap_step, brain_mask_glo_img)
        logger.debug('number of patches generated in %s is %s' % (data['id'], len(starts_list)))
        for starts in starts_list:
            yield _gen_patch(starts)


def get_sliding_window_patch_starts(input_img: np.ndarray, patch_size, overlap_step, reference_mask=None):
    assert input_img.ndim == 3
    d, h, w = input_img.shape
    d_starts = list(range(0, d - patch_size[0], overlap_step[0])) + [d - patch_size[0]]
    h_starts = list(range(0, h - patch_size[1], overlap_step[1])) + [h - patch_size[1]]
    w_starts = list(range(0, w - patch_size[2], overlap_step[2])) + [w - patch_size[2]]

    patch_starts = []
    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                if reference_mask is None:
                    patch_starts.append([ds, hs, ws])
                else:
                    if reference_mask[ds:ds + patch_size[0], hs:hs + patch_size[1], ws:ws + patch_size[2]].sum() \
                            > 0.05 * patch_size[0] * patch_size[1] * patch_size[2]:
                        patch_starts.append([ds, ws, hs])  # only compute patches who have more than 5% reference mask
    return patch_starts


def get_positive_region_centers(label, return_object_wise_label=False):
    label = measure.label(label)
    pros = measure.regionprops(label)
    centers = [c.centroid for c in pros if c.area > 5]  # ignore small noise region
    if return_object_wise_label:
        return centers, label
    else:
        return centers


def resize_image(image, old_spacing=None, new_spacing=None, new_shape=None, order=1):
    assert new_shape is not None or (old_spacing is not None and new_spacing is not None)
    if new_shape is None:
        new_shape = tuple([int(np.round(old_spacing[i] / new_spacing[i] * float(image.shape[i]))) for i in range(3)])
    resized_image = resize(image, new_shape, order=order, mode='edge', cval=0, anti_aliasing=False)
    return resized_image


def resize_segmentation(segmentation, old_spacing=None, new_spacing=None, new_shape=None, order=0, cval=0):
    '''
    Taken from batchgenerators (https://github.com/MIC-DKFZ/batchgenerators) to prevent dependency

    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    assert new_shape is not None or (old_spacing is not None and new_spacing is not None)
    if new_shape is None:
        new_shape = tuple(
            [int(np.round(old_spacing[i] / new_spacing[i] * float(segmentation.shape[i]))) for i in range(3)])
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation, new_shape, order, mode="constant", cval=cval, clip=True,
                      anti_aliasing=False).astype(tpe)
    else:
        unique_labels = np.unique(segmentation)
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        for i, c in enumerate(unique_labels):
            reshaped_multihot = resize((segmentation == c).astype(float), new_shape, order, mode="edge", clip=True,
                                       anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


class GlobalLocalizer:
    def __init__(self, reference_mask):
        assert reference_mask.ndim == 3
        self.original_shape = reference_mask.shape
        self.mask = reference_mask
        starts = [0, 0, 0]
        ends = list(self.mask.shape)
        while starts[0] < self.original_shape[0]:
            if self.mask[starts[0], :, :].sum() > 0:
                break
            starts[0] += 1
        while ends[0] > starts[0]:
            if self.mask[ends[0] - 1, :, :].sum() > 0:
                break
            ends[0] -= 1
        while starts[1] < self.original_shape[1]:
            if self.mask[:, starts[1], :].sum() > 0:
                break
            starts[1] += 1
        while ends[1] > starts[1]:
            if self.mask[:, ends[1] - 1, :].sum() > 0:
                break
            ends[1] -= 1
        while starts[2] < self.original_shape[2]:
            if self.mask[:, :, starts[2]].sum() > 0:
                break
            starts[2] += 1
        while ends[2] > starts[2]:
            if self.mask[:, :, ends[2] - 1].sum() > 0:
                break
            ends[2] -= 1

        self.starts = starts
        self.ends = ends

    def cut_edge(self, img, new_shape=None, is_mask=False):
        assert img.shape == self.original_shape
        cut_img = img[self.starts[0]:self.ends[0], self.starts[1]:self.ends[1], self.starts[2]:self.ends[2]].copy()
        if new_shape is not None:
            cut_img = self.reshape_keep_ratio(cut_img, new_shape, is_mask)
        return cut_img

    def get_cut_reference_mask(self, new_shape=None):
        return self.cut_edge(self.mask, new_shape, is_mask=True)

    def reshape_keep_ratio(self, img, new_shape, is_mask=False):
        assert len(new_shape) == 3
        ori_shape = img.shape
        rel_index = np.argmin(np.array([new_shape[i] / ori_shape[i] for i in range(3)]))
        pad_shape = [round(ori_shape[rel_index] * new_shape[i] / new_shape[rel_index]) for i in range(3)]
        padded_img = np.pad(img, tuple([((pad_shape[i] - ori_shape[i]) // 2,) for i in range(3)]),
                            mode='constant', constant_values=img.min())
        if is_mask:
            new_img = resize_segmentation(padded_img, new_shape=new_shape)
        else:
            new_img = resize_image(padded_img, new_shape=new_shape)
        return new_img

    def get_position_map(self, starts, ends, new_shape=None):
        position_map = np.zeros(self.original_shape, np.float32)
        position_map[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] = 1
        cut_position_map = self.cut_edge(position_map)
        if new_shape is not None:
            cut_position_map = self.reshape_keep_ratio(cut_position_map, new_shape, is_mask=True)
        return cut_position_map

    def get_position_bbox(self, starts, ends, new_shape=None):
        if new_shape is None:
            new_shape = self.original_shape
        reference_index = np.argmin(np.array([new_shape[i] / self.original_shape[i] for i in range(3)]))
        new_starts = [min(max(0, round((
                                               2 * starts[i] * new_shape[reference_index] + new_shape[i] *
                                               self.original_shape[reference_index] -
                                               self.original_shape[i] * new_shape[reference_index]) / (
                                                   2 * self.original_shape[reference_index]))), new_shape[i] - 1) for i
                      in
                      range(3)]
        new_ends = [min(max(new_starts[i], round((
                                                         2 * ends[i] * new_shape[reference_index] + new_shape[i] *
                                                         self.original_shape[reference_index] -
                                                         self.original_shape[i] * new_shape[
                                                             reference_index]) / (
                                                         2 * self.original_shape[reference_index])) - 1),
                        new_shape[i] - 1) for i in
                    range(3)]
        new_bbox = np.array([new_starts[0], new_ends[0], new_starts[1], new_ends[1], new_starts[2], new_ends[2]])
        return new_bbox


def random_flip_all(img, do_it=(None, None, None)):
    img = random_flip(img, 1, do_it[0])
    img = random_flip(img, 2, do_it[1])
    img = random_flip(img, 3, do_it[2])
    return img


def random_rotate_all(img, do_it=(None, None, None)):
    img = random_rotate(img, 1, do_it[0])
    img = random_rotate(img, 2, do_it[1])
    img = random_rotate(img, 3, do_it[2])
    return img


def random_flip(img, dim, do_it=None):
    assert len(img.shape) == 4  # c, d, w, h
    assert dim in [1, 2, 3]
    norm_img = img

    if do_it is None:
        if np.random.rand() > 0.5:
            do_it = False
        else:
            do_it = True
    if do_it:
        out_img = np.flip(norm_img, [dim])
    else:
        out_img = norm_img
    return out_img


def random_rotate(img, dim, do_it=None):
    assert len(img.shape) == 4  # c, d, w, h
    assert dim in [1, 2, 3]

    norm_img = img

    if dim == 1:
        perm = [0, 1, 3, 2]
    elif dim == 2:
        perm = [0, 3, 2, 1]
    else:
        perm = [0, 2, 1, 3]

    if do_it is None:
        if np.random.rand() > 0.5:
            do_it = True
        else:
            do_it = False
    if do_it:
        out_img = np.transpose(norm_img, perm)
    else:
        out_img = norm_img
    return out_img


def get_instances_from_file_or_folder(instance_file_or_folder, instance_type='nii', drop_phrase=None,
                                      require_phrase=None):
    assert instance_type in ['nii', 'dcm']
    assert os.path.exists(instance_file_or_folder)

    if drop_phrase is not None:
        if not (isinstance(drop_phrase, list) or isinstance(drop_phrase, tuple)):
            drop_phrase = [drop_phrase]
    if require_phrase is not None:
        if not (isinstance(require_phrase, list) or isinstance(require_phrase, tuple)):
            require_phrase = [require_phrase]

    instances = []
    if instance_type == 'nii':
        if os.path.isdir(instance_file_or_folder):
            for ins in os.listdir(instance_file_or_folder):
                if ins.endswith('.nii.gz'):
                    if drop_phrase is None or all([dp not in ins for dp in drop_phrase]):
                        if require_phrase is None or all([rp in ins for rp in require_phrase]):
                            instances.append(os.path.join(instance_file_or_folder, ins))
        else:
            if instance_file_or_folder.endswith('.nii.gz'):
                if drop_phrase is None or all([dp not in instance_file_or_folder for dp in drop_phrase]):
                    if require_phrase is None or all([rp in instance_file_or_folder for rp in require_phrase]):
                        instances.append(instance_file_or_folder)
    else:
        reader = sitk.ImageSeriesReader()

        def _find_base(_folder, _ins):
            is_base = True
            for item in os.listdir(_folder):
                if os.path.isdir(os.path.join(_folder, item)):
                    is_base = False
                    _ins = _find_base(os.path.join(_folder, item), _ins)
            if is_base:
                dcm_files = reader.GetGDCMSeriesFileNames(_folder)
                _ins.append(dcm_files)
            return _ins

        if os.path.isdir(instance_file_or_folder):
            instances = _find_base(instance_file_or_folder, instances)
        else:
            instances = _find_base(os.path.dirname(instance_file_or_folder), instances)
    return instances
