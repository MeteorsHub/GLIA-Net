import argparse
import os
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np
import torch

from data_loader import get_instances_from_file_or_folder
from utils.metrics import get_evaluation_metric
from utils.project_utils import str2bool, get_devices, get_logger, load_config

parser = argparse.ArgumentParser(description='AneurysmSeg study evaluation')
parser.add_argument('-c', '--config', type=str, required=False, default='eval_per_case',
                    help='config name. default: \'study_evaluate\'')
parser.add_argument('-d', '--device', type=str, required=False, default='0',
                    help='device id for cuda and \'cpu\' for cpu. can be multiple devices split by \',\'.')
parser.add_argument('-g', '--gt_file_or_folder', type=str, required=True,
                    help='ground truth file or folder to be the gt segmentation mask')
parser.add_argument('-p', '--pred_file_or_folder', type=str, required=False,
                    help='prediction file or folder to be the prediction segmentation mask or probability distribution')
parser.add_argument('-l', '--logging_folder', type=str, required=False, default=None,
                    help='where to put the logging messages.')
parser.add_argument('-m', '--mask', type=str2bool, default='true', required=False,
                    help='If prediction file or folder is segmentation mask. Else probability distribution. ')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='whether to use verbose/debug logging level.')
args = parser.parse_args()


def study_evaluate(config, gt_file_or_folder, pred_file_or_folder, logger, devices, pred_type='mask'):
    assert pred_type in ['mask', 'prob']
    logger.info('use device %s' % args.device)

    logger.info('gt_file_or_folder: %s' % gt_file_or_folder)
    logger.info('pred_file_or_folder: %s' % pred_file_or_folder)
    logger.info('mask or probability distribution: %s' % pred_type)
    if pred_type == 'prob':
        logger.info('threshold: %1.2f' % config['eval'].get('probability_threshold', 0.5))
        drop_phrase = None
        require_phrase = '_prob'
    else:
        drop_phrase = '_prob'
        require_phrase = None
    logger.info('Begin to scan gt_folder_or_file %s...' % gt_file_or_folder)
    gt_instances = sorted(get_instances_from_file_or_folder(gt_file_or_folder, 'nii', drop_phrase='_pred'))
    logger.info('Begin to scan pred_folder_or_file %s...' % pred_file_or_folder)
    pred_instances = sorted(get_instances_from_file_or_folder(pred_file_or_folder, 'nii', drop_phrase, require_phrase))

    if len(gt_instances) != len(pred_instances):
        logger.critical('numbers of gt_instances and pred_instances do not match: %d, %d'
                        % (len(gt_instances), len(pred_instances)))
        exit(1)
    else:
        logger.info('instance number: %d. start evaluating...' % len(gt_instances))

    eval_metric_fns, eval_curve_fns = get_evaluation_metric(config, logger, devices[0])
    for metric_fn in eval_metric_fns.values():
        metric_fn.reset()

    reader = sitk.ImageFileReader()
    for i, (gt_ins, pred_ins) in enumerate(zip(gt_instances, pred_instances)):
        ins_id = os.path.basename(gt_ins).split('.')[0]
        reader.SetFileName(gt_ins)
        gt_img = reader.Execute()
        gt_img = sitk.GetArrayFromImage(gt_img).astype(np.int32)
        reader.SetFileName(pred_ins)
        pred_img = reader.Execute()
        pred_img = sitk.GetArrayFromImage(pred_img).astype(np.float32)

        gt_img = torch.unsqueeze(torch.tensor(gt_img, dtype=torch.int8, device=devices[0]), 0)  # [b, ...]
        pred_img = torch.unsqueeze(torch.tensor(pred_img, dtype=torch.float32, device=devices[0]), 0)
        pred_img = torch.stack([1.0 - pred_img, pred_img], 1)  # [b, c, ...]

        current_metrics = OrderedDict()
        depth = pred_img.shape[2]
        if pred_img.shape[2] > 500:
            for key, metric_fn in eval_metric_fns.items():
                current_metrics[key] = metric_fn(pred_img[:, :, :depth // 2], gt_img[:, :depth // 2])
                current_metrics[key] = metric_fn(pred_img[:, :, depth // 2:], gt_img[:, depth // 2:])
                if isinstance(current_metrics[key], float):
                    current_metrics[key] = current_metrics[key] / 2
        else:
            for key, metric_fn in eval_metric_fns.items():
                current_metrics[key] = metric_fn(pred_img, gt_img)

        logging_info = '(%d in %d) %s:' % (i + 1, len(gt_instances), ins_id)
        for metric_name, metric_value in current_metrics.items():
            if isinstance(metric_value.item(), int):
                logging_info += '\t%s: %d' % (metric_name, metric_value.item())
            else:
                logging_info += '\t%s: %1.4f' % (metric_name, metric_value.item())
        logger.info(logging_info)

    logging_info = 'overall:'
    for metric_name, metric_fn in eval_metric_fns.items():
        if isinstance(metric_fn.result.item(), int):
            logging_info += '\t%s: %d' % (metric_name, metric_fn.result.item())
        else:
            logging_info += '\t%s: %1.4f' % (metric_name, metric_fn.result.item())
    logger.info(logging_info)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    if args.logging_folder is None:
        logging_folder = args.pred_file_or_folder
    else:
        logging_folder = args.logging_folder
    logger = get_logger('StudyEvaluater', logging_folder, args.verbose, logging_prefix='evaluate_study')
    ori_config = load_config(os.path.join('configs', args.config + '.yaml'))
    config = OrderedDict()
    config['model'] = {'num_classes': 2}
    if args.mask:
        config['eval'] = ori_config['eval_mask']
    else:
        config['eval'] = ori_config['eval_prob']
    devices = get_devices(args.device, logger)
    try:
        pred_type = 'mask' if args.mask else 'prob'
        study_evaluate(config, args.gt_file_or_folder, args.pred_file_or_folder, logger, devices, pred_type)
    except Exception as e:
        logger.exception(e)
