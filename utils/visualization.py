import importlib
import random

import numpy as np
import torch
import torchvision.utils
from PIL import Image, ImageDraw


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (inputs/outputs to the network or the target segmentation
    images) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, num_slices_per_cube=5, center_depth_step=2, **kwargs):
        self.num_slices_per_cube = num_slices_per_cube
        self.center_depth_step = center_depth_step

    def __call__(self, img_dict, max_num_samples=1):
        """
        Transform a batch of images to one summary image in the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.
        Args:
             img_dict: name, tensor dict with input, target and pred or some of them. Tensor dim must be 5
             max_num_samples: how many samples in the batch will be selected
        """
        assert len(img_dict) > 0
        for key in img_dict.keys():
            assert isinstance(img_dict[key], np.ndarray)
            assert img_dict[key].ndim == 5
            assert img_dict[key].shape[1] == 1 or img_dict[key].shape[1] == 3
            if img_dict[key].shape[1] == 1:
                img_dict[key] = np.repeat(img_dict[key], 3, 1)
        img_shape = list(img_dict.values())[0].shape
        for value in img_dict.values():
            assert value.shape == img_shape, 'all the img shapes should be the same'
        tagged_image = self.process_batch(img_dict, max_num_samples)

        def _check_img(tag_img):
            tag, img = tag_img
            if img is None:
                return tag, img  # no img summary

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                assert img.shape[0] or img.shape[0], 'Only (1, H, W) or (3, H, W) images are supported'
            return tag, img
        _check_img(tagged_image)

        return tagged_image

    def process_batch(self, img_dict):
        """
        Sub implementation
        :param img_dict: {name: ndarray[b, c, d, h, w],...}
        :return: tag, img
        """
        raise NotImplementedError


class GridTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, num_slices_per_cube=5, center_depth_step=2, **kwargs):
        super(GridTensorboardFormatter, self).__init__(num_slices_per_cube, center_depth_step, **kwargs)

    def process_batch(self, img_dict, max_num_samples=1):
        """img_dict should be normalized to 0-1"""
        tag = 'OOOO'.join(list(img_dict.keys()))
        img_tags = list(img_dict.keys())
        imgs = list(img_dict.values())
        batch, channel, depth, height, width = imgs[0].shape
        target_positive_center_d = None
        target_img = None
        default_depth_index = np.linspace(0, depth, self.num_slices_per_cube + 2, dtype=np.int8)[1:-1]

        # find depth_indexes to log
        for i in range(len(img_dict)):
            if 'target' in img_tags[i]:
                target_positive_center_d = np.mean(np.argmax(np.sum(imgs[i], (3, 4)), 2), 1).astype(np.int8)
                target_img = imgs[i]

        depth_indexes = []
        selected_batch_img_indexes = []
        # has target images
        if target_positive_center_d is not None:
            for i, v in enumerate(target_positive_center_d):  # iter among each img in the batch
                # no positive targets
                if np.sum(target_img[i]) < 3:
                    if random.random() < 0.5:  # 50% negative regions are saved
                        depth_indexes.append(default_depth_index)
                        selected_batch_img_indexes.append(i)
                    else:
                        continue
                # full of positive targets
                elif np.mean(target_img[i] > 0.95):
                    depth_indexes.append(default_depth_index)
                    selected_batch_img_indexes.append(i)
                # one or a few positive targets
                else:
                    depth_index = [v]
                    left = True
                    for _ in range(self.num_slices_per_cube - 1):
                        # boundary
                        if depth_index[0] - self.center_depth_step < 0:
                            left = False
                        if depth_index[-1] + self.center_depth_step >= depth:
                            left = True
                        if left:
                            depth_index = [depth_index[0] - self.center_depth_step] + depth_index
                        else:
                            depth_index += [depth_index[-1] + self.center_depth_step]
                        left = not left
                    depth_indexes.append(np.array(depth_index, np.int8))
                    selected_batch_img_indexes.append(i)
        # no target images
        else:
            depth_indexes = np.tile(np.expand_dims(default_depth_index, 0), (batch, 1))
            selected_batch_img_indexes = list(range(batch))

        if max_num_samples < len(depth_indexes):
            depth_indexes = depth_indexes[:max_num_samples]
            selected_batch_img_indexes = selected_batch_img_indexes[:max_num_samples]
        if len(selected_batch_img_indexes) == 0:
            return tag, None
        log_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][selected_batch_img_indexes]
            img = []
            for j in range(len(selected_batch_img_indexes)):
                img.append(imgs[i][j, :, depth_indexes[j], ...])
            img = np.stack(img)
            log_imgs.append(img)
        log_imgs = np.stack(log_imgs)
        log_imgs = np.transpose(log_imgs, [1, 0, 2, 3, 4, 5])
        log_imgs = np.reshape(log_imgs, [-1, channel, height, width])

        log_imgs = torchvision.utils.make_grid(torch.tensor(log_imgs), self.num_slices_per_cube)
        return tag, log_imgs


def get_tensorboard_formatter(formatter_name, **kwargs):
    module = importlib.import_module('utils.visualization')
    return getattr(module, formatter_name)(**kwargs)


def get_text_image(text, image_h, image_w, norm=True):
    img = Image.new('RGB', (image_w, image_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = draw.textsize(text, )
    draw.text(((image_w - w) / 2, (image_h - h) / 2), text, fill="black")
    img_arr = np.array(img)
    if norm:
        img_arr = img_arr.astype(np.float32) / 255
    return img_arr
