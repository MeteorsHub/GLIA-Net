import argparse
import os

import SimpleITK as sitk
import numpy as np
import skimage.measure as measure
import torch

from data_loader import get_instances_from_file_or_folder
from utils.project_utils import get_devices, get_logger, maybe_create_path

parser = argparse.ArgumentParser(description='AneurysmSeg refinement with morph close and small or thin target removal')
parser.add_argument('-d', '--device', type=str, required=False, default='0',
                    help='device id for cuda and \'cpu\' for cpu. can be multiple devices split by \',\'.')
parser.add_argument('-i', '--input_file_or_folder', type=str, required=True,
                    help='prediction file or folder to be the label mask or probability map prediction')
parser.add_argument('-o', '--output_file_or_folder', type=str, required=True,
                    help='where to store new files')
parser.add_argument('-k', '--kernel_size', type=int, required=False, default=7,
                    help='kernel size in morph close')
parser.add_argument('-a', '--area_threshold', type=int, required=False, default=30,
                    help='target whose area is no more than this threshold will be dropped')
parser.add_argument('-t', '--thin_threshold', type=int, required=False, default=1,
                    help='target who has no more than this number of slices along any axis will be dropped')
parser.add_argument('-l', '--logging_folder', type=str, required=False, default=None,
                    help='where to put the logging messages.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='whether to use verbose/debug logging level.')
args = parser.parse_args()


def refine_segmentation(input_file_or_folder, output_file_or_folder, kernel_size, area_threshold, thin_threshold,
                        devices, logger):
    logger.info('use device %s' % args.device)
    logger.info('input_file_or_folder: %s' % input_file_or_folder)
    logger.info('output_file_or_folder: %s' % output_file_or_folder)

    logger.info('Begin to scan input_folder_or_file %s...' % input_file_or_folder)
    input_instances = sorted(get_instances_from_file_or_folder(input_file_or_folder, 'nii', drop_phrase=None))
    logger.info('instance number: %d. kernel_size: %d. area_threshold: %d. thin_threshold: %d. start processing...' % (
    len(input_instances), kernel_size, area_threshold, thin_threshold))
    if os.path.isdir(input_file_or_folder):
        maybe_create_path(output_file_or_folder)
    else:
        maybe_create_path(os.path.dirname(output_file_or_folder))

    reader = sitk.ImageFileReader()
    for i, input_ins in enumerate(input_instances):
        ins_id = os.path.basename(input_ins).split('.')[0]
        reader.SetFileName(input_ins)
        input_itk_img = reader.Execute()
        input_img = sitk.GetArrayFromImage(input_itk_img).astype(np.int32)
        _, input_label_num = measure.label(input_img, return_num=True)

        # morph close
        morph_close_img = torch.tensor(input_img, device=devices[i % len(devices)])
        morph_close_img = torch.unsqueeze(torch.unsqueeze(morph_close_img, 0), 0).type(torch.float32)
        padding = kernel_size // 2
        # Dilated
        morph_close_img = torch.nn.MaxPool3d(kernel_size, stride=1, padding=padding)(morph_close_img)
        # Eroded
        morph_close_img = 1.0 - torch.nn.MaxPool3d(kernel_size, stride=1, padding=padding)(1.0 - morph_close_img)
        morph_close_img = torch.squeeze(torch.squeeze(morph_close_img, 0), 0).type(torch.int32)

        morph_close_img = morph_close_img.detach().cpu().numpy()

        # remove small or thin targets
        morph_close_label, morph_close_label_num = measure.label(morph_close_img, return_num=True)
        morph_close_props = measure.regionprops(morph_close_label)
        output_label = morph_close_label.copy()
        remove_small_count = 0
        remove_thin_count = 0
        for prop in morph_close_props:
            if prop.area <= area_threshold:
                output_label = np.where(output_label == prop.label,
                                        np.zeros_like(output_label),
                                        output_label)
                remove_small_count += 1
            else:
                for j in range(len(prop.bbox) // 2):
                    if prop.bbox[j + len(prop.bbox) // 2] - prop.bbox[j] <= thin_threshold:
                        output_label = np.where(output_label == prop.label,
                                                np.zeros_like(output_label),
                                                output_label)
                        remove_thin_count += 1
                        break
        output_img = (output_label > 0).astype(np.int32)

        if os.path.isdir(input_file_or_folder):
            output_file = os.path.join(output_file_or_folder, os.path.basename(input_ins))
        else:
            output_file = output_file_or_folder
        output_itk_image = sitk.GetImageFromArray(output_img)
        output_itk_image.CopyInformation(input_itk_img)
        sitk.WriteImage(output_itk_image, output_file)

        logging_info = '(%d in %d) targets number of %s: %d -> morph close -> %d -> remove small -> %d -> remove thin -> %d.' \
                       % (i + 1, len(input_instances), ins_id, input_label_num, morph_close_label_num,
                          morph_close_label_num - remove_small_count,
                          morph_close_label_num - remove_small_count - remove_thin_count)
        logger.info(logging_info)


if __name__ == '__main__':
    if args.logging_folder is None:
        logging_folder = args.input_file_or_folder
    else:
        logging_folder = args.logging_folder
    logger = get_logger('RefineSegmentation', logging_folder, args.verbose, logging_prefix='refine_segmentation')
    devices = get_devices(args.device, logger)
    try:
        refine_segmentation(args.input_file_or_folder, args.output_file_or_folder, args.kernel_size,
                            args.area_threshold, args.thin_threshold, devices, logger)
    except Exception as e:
        logger.exception(e)
