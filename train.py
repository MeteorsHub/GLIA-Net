import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data

from core import Trainer
from data_loader import AneurysmSegDataset, TaskListQueue
from utils.project_utils import load_config, get_logger, save_config, get_devices

parser = argparse.ArgumentParser(description='AneurysmSeg training')
parser.add_argument('-c', '--config', type=str, required=False, default='default',
                    help='config name. default: \'default\'')
parser.add_argument('-n', '--exp_id', type=int, required=False, default=1,
                    help='to identify different exp ids.')
parser.add_argument('-d', '--device', type=str, required=False, default='0',
                    help='device id for cuda and \'cpu\' for cpu. can be multiple devices split by \',\'.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='whether to use verbose/debug logging level.')
parser.add_argument('-o', '--overwrite_config', action='store_true',
                    help='if true, ignore existing config file in exp path and use that in config folder')
args = parser.parse_args()


def train(config, exp_path, logger, devices):
    if config['train'].get('manual_seed') is not None:
        manual_seed = config['train'].get('manual_seed')
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
            # see https://pytorch.org/docs/stable/notes/randomness.html
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(manual_seed)

    train_task_list_queue = TaskListQueue(config, 'train', logger, config['data']['train_num_file'], shuffle_files=True)
    train_dataset = AneurysmSegDataset(config, 'train', train_task_list_queue, logger)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['train']['batch_size'] * len(devices), drop_last=True,
        num_workers=config['data']['num_proc_workers'])
    if config['train']['eval_every_n_epochs'] != 0:
        eval_task_list_queue = TaskListQueue(config, 'eval', logger, config['data']['eval_num_file'],
                                             shuffle_files=False)
        eval_dataset = AneurysmSegDataset(config, 'eval', eval_task_list_queue, logger)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config['train']['batch_size'] * len(devices),
                                                  num_workers=config['data']['num_proc_workers'], drop_last=True)
    else:
        eval_loader = None

    trainer = Trainer(config, exp_path, devices, train_loader, eval_loader, logger=logger)
    trainer.train()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    exp_path = os.path.join('exp', args.config)
    if not args.overwrite_config and os.path.exists(os.path.join(exp_path, 'config.yaml')):
        config = load_config(os.path.join(exp_path, 'config.yaml'))
    else:
        config = load_config(os.path.join('configs', args.config + '.yaml'))
        save_config(os.path.join(exp_path, 'config.yaml'), config)

    exp_path = os.path.join(exp_path, str(args.exp_id))

    logging_folder = os.path.join(exp_path, config.get('logging_folder')) \
        if config.get('logging_folder') is not None else None
    logger = get_logger('Task%sTrainer' % config['task'], logging_folder, args.verbose)
    logger.debug('config loaded:\n%s', config)
    devices = get_devices(args.device, logger)
    logger.info('use device %s' % args.device)

    try:
        train(config, exp_path, logger, devices)
    except Exception as e:
        logger.exception(e)
