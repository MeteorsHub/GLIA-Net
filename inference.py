import argparse
import os

from core import Inferencer
from data_loader import AneurysmSegTestManager
from utils.project_utils import load_config, get_logger, get_devices, str2bool

parser = argparse.ArgumentParser(description='AneurysmSeg evaluation')
parser.add_argument('-c', '--config', type=str, required=False, default='default',
                    help='config name. default: \'default\'')
parser.add_argument('-n', '--exp_id', type=int, required=False, default=1,
                    help='to identify different exp ids.')
parser.add_argument('-d', '--device', type=str, required=False, default='0',
                    help='device id for cuda and \'cpu\' for cpu. can be multiple devices split by \',\'.')
parser.add_argument('-i', '--input_file_or_folder', type=str, required=True,
                    help='input file or folder to be the input image(s)')
parser.add_argument('-t', '--input_type', choices=['nii', 'dcm'], default='nii', required=False,
                    help='nii or dicom file type')
parser.add_argument('-o', '--output_folder', type=str, default=None, required=False,
                    help='where to save the output, default is to save to the same folder as input')
parser.add_argument('-b', '--save_binary', type=str2bool, default='true', required=False,
                    help='whether to save label mask predictions.')
parser.add_argument('-p', '--save_prob', type=str2bool, default='false', required=False,
                    help='whether to save probability map predictions')
parser.add_argument('-g', '--save_global', type=str2bool, default='false', required=False,
                    help='whether to save global positioning outputs.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='whether to use verbose/debug logging level.')
args = parser.parse_args()


def inference(config, exp_path, logger, devices, input_file_or_folder, input_type, output_folder, save_binary,
              save_prob, save_global):
    inference_data_manager = AneurysmSegTestManager(config, logger, devices)
    inferencer = Inferencer(config, exp_path, devices, input_file_or_folder, output_folder, input_type, save_binary,
                            save_prob, save_global, inference_data_manager, logger)
    inferencer.inference()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    exp_path = os.path.join('exp', args.config.replace('inference_', ''))
    config = load_config(os.path.join('configs', args.config + '.yaml'))

    exp_path = os.path.join(exp_path, str(args.exp_id))

    logging_folder = os.path.join(exp_path, config.get('logging_folder')) \
        if config.get('logging_folder') is not None else None
    logger = get_logger('Task%sInferencer' % config['task'], logging_folder, args.verbose)
    logger.debug('config loaded:\n%s', config)
    devices = get_devices(args.device, logger)
    logger.info('use device %s' % args.device)

    try:
        inference(config, exp_path, logger, devices, args.input_file_or_folder, args.input_type, args.output_folder,
                  args.save_binary, args.save_prob, args.save_global)
    except Exception as e:
        logger.exception(e)
