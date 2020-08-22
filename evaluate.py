import argparse
import os

import numpy as np
import torch

from core import Evaler
from data_loader import AneurysmSegDataset, TaskListQueue
from utils.project_utils import load_config, get_logger, get_devices

parser = argparse.ArgumentParser(description='AneurysmSeg evaluation')
parser.add_argument('-c', '--config', type=str, required=False, default='default',
                    help='config name. default: \'default\'')
parser.add_argument('-n', '--exp_id', type=int, required=False, default=1,
                    help='to identify different exp ids.')
parser.add_argument('-d', '--device', type=str, required=False, default='0',
                    help='device id for cuda and \'cpu\' for cpu. can be multiple devices split by \',\'.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='whether to use verbose/debug logging level.')
args = parser.parse_args()


def eval(config, exp_path, logger, devices):
    if config['train'].get('manual_seed') is not None:
        manual_seed = config['train'].get('manual_seed')
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
            # see https://pytorch.org/docs/stable/notes/randomness.html
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(manual_seed)

    eval_task_list_queue = TaskListQueue(config, 'eval', logger, config['data']['eval_num_file'], shuffle_files=False)
    eval_dataset = AneurysmSegDataset(config, 'eval', eval_task_list_queue, logger)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config['train']['batch_size'] * len(devices),
                                              num_workers=config['data']['num_proc_workers'], drop_last=True)

    evaler = Evaler(config, exp_path, devices, eval_loader, logger=logger)
    evaler.eval()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    exp_path = os.path.join('exp', args.config.replace('eval_', ''))
    config = load_config(os.path.join('configs', args.config + '.yaml'))

    exp_path = os.path.join(exp_path, str(args.exp_id))

    logging_folder = os.path.join(exp_path, config.get('logging_folder')) \
        if config.get('logging_folder') is not None else None
    logger = get_logger('Task%sEvaler' % config['task'], logging_folder, args.verbose)
    logger.debug('config loaded:\n%s', config)
    devices = get_devices(args.device, logger)
    logger.info('use device %s' % args.device)

    try:
        eval(config, exp_path, logger, devices)
    except Exception as e:
        logger.exception(e)
