import glob
import importlib
import logging
import os

import numpy as np
import torch


def get_model(config, logger: logging.Logger):
    def _model_class(model_conf):
        m = importlib.import_module('model.' + model_conf['filename'])
        m = getattr(m, model_conf['classname'])
        return m

    if 'model' not in config:
        logger.critical('Could not find model configuration')
        exit(1)
    model_config = config['model']
    model_class = _model_class(model_config)
    return model_class(**model_config)


class CheckpointLogger:
    def __init__(self, checkpoint_dir, logger, max_keep_runs=None):
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        if max_keep_runs is None:
            self.max_keep_runs = float('inf')
            self.max_saved = 5
        else:
            self.max_keep_runs = max_keep_runs
            self.max_saved = max_keep_runs // 4
        if not os.path.exists(checkpoint_dir):
            logger.info(
                f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
            os.mkdir(checkpoint_dir)
        self.has_saved = 0

    def get_latest_ckpt_file_list(self):
        latest_file_list = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint-???????.pt'))
        latest_file_list = sorted(latest_file_list, key=lambda x: int(x[-10:-3]), reverse=False)
        while len(latest_file_list) > self.max_keep_runs:
            os.remove(latest_file_list.pop())
        return latest_file_list

    def get_best_ckpt_file(self):
        best_file = glob.glob(os.path.join(self.checkpoint_dir, 'best_checkpoint-*.pt'))
        if best_file:
            return best_file[0]
        else:
            return None

    def is_ckpt_exist(self, is_best=False):
        if is_best:
            if self.get_best_ckpt_file() is None:
                return False
        else:
            if len(self.get_latest_ckpt_file_list()) == 0:
                return False
        return True

    def save_checkpoint(self, state, is_best):
        if is_best:
            best_file = self.get_best_ckpt_file()
            if best_file is not None:
                os.remove(best_file)
            best_file = os.path.join(self.checkpoint_dir, 'best_checkpoint-%1.4f.pt' % state['best_eval_score'])
            torch.save(state, best_file)
            self.logger.info(f"Best checkpoint saved to '{best_file}'")
        else:
            latest_ckpt_list = self.get_latest_ckpt_file_list()
            if len(latest_ckpt_list) == self.max_keep_runs:
                if self.has_saved == 0:
                    os.remove(latest_ckpt_list.pop(0))
                else:
                    os.remove(latest_ckpt_list.pop(-1 - self.has_saved))
                    self.has_saved -= 1
            if self.has_saved == self.max_saved:
                os.remove(latest_ckpt_list.pop(-1 - self.has_saved))
                self.has_saved -= 1
            latest_ckpt_file = os.path.join(self.checkpoint_dir, 'checkpoint-%07d.pt' % state['num_iteration'])
            torch.save(state, latest_ckpt_file)
            self.logger.info(f"Latest checkpoint saved to '{latest_ckpt_file}'")
            self.has_saved += 1

    def load_ckeckpoint(self, model, ckpt_file=None, is_best=False, optimizer=None):
        if ckpt_file is None:
            if is_best:
                ckpt_file = self.get_best_ckpt_file()
            else:
                ckpt_files = self.get_latest_ckpt_file_list()
                if len(ckpt_files) == 0:
                    self.logger.warning("Checkpoint does not exist. Performing as from scratch")
                    return
                ckpt_file = ckpt_files[-1]
        else:
            ckpt_file = os.path.join(self.checkpoint_dir, ckpt_file)

        state = torch.load(ckpt_file, map_location=lambda storage, location: storage)
        model.load_state_dict(state['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        self.logger.info(f"Checkpoint '{ckpt_file}' loaded")
        return state


def create_optimizer(config, model):
    if 'optimizer' not in config:
        return None
    optimizer_config = config['optimizer']
    optimizer_cls = getattr(torch.optim, optimizer_config['name'])
    optimizer = optimizer_cls(model.parameters(), **optimizer_config['kwargs'])
    return optimizer


def create_lr_scheduler(config, optimizer):
    if 'lr_scheduler' not in config:
        return None
    lr_config = config.get('lr_scheduler')
    lr_scheduler_cls = getattr(torch.optim.lr_scheduler, lr_config['name'])
    return lr_scheduler_cls(optimizer, **lr_config['kwargs'])


def binary_mask2bbox(binary_mask):
    assert isinstance(binary_mask, np.ndarray) or isinstance(binary_mask, torch.Tensor)
    if binary_mask.ndim == 5:
        assert binary_mask.shape[1] == 1
        if isinstance(binary_mask, torch.Tensor):
            binary_mask = torch.squeeze(binary_mask, 1)
        else:
            binary_mask = np.squeeze(binary_mask, 1)
    if binary_mask.ndim == 4:
        if isinstance(binary_mask, torch.Tensor):
            return torch.stack([binary_mask2bbox(binary_mask[i] for i in range(len(binary_mask)))])
        else:
            return np.stack([binary_mask2bbox(binary_mask[i] for i in range(len(binary_mask)))])
    assert binary_mask.ndim == 3
    if isinstance(binary_mask, torch.Tensor):
        device = binary_mask.device
        binary_mask = binary_mask.detach().cpu().numpy()
    else:
        device = None
    d_mask = (binary_mask.sum((1, 2)) > 0).astype(np.float32)
    d_start = int(np.argmax(d_mask))
    d_end = int(d_start + d_mask.sum() - 1)
    h_mask = (binary_mask.sum((0, 2)) > 0).astype(np.float32)
    h_start = int(np.argmax(h_mask))
    h_end = int(h_start + h_mask.sum() - 1)
    w_mask = (binary_mask.sum((0, 1)) > 0).astype(np.float32)
    w_start = int(np.argmax(w_mask))
    w_end = int(w_start + w_mask.sum() - 1)
    bbox = np.array([d_start, d_end, h_start, h_end, w_start, w_end], np.int32)
    if device is not None:
        bbox = torch.from_numpy(bbox).to(device)
    return bbox


def bbox2binary_mask(bbox, img_shape):
    assert isinstance(bbox, torch.Tensor) or isinstance(bbox, np.ndarray)
    assert len(img_shape) == 3
    if bbox.ndim == 2:
        if isinstance(bbox, torch.Tensor):
            return torch.stack([bbox2binary_mask(bbox[i], img_shape) for i in range(len(bbox))])
        else:
            return np.stack([bbox2binary_mask(bbox[i], img_shape) for i in range(len(bbox))])
    assert bbox.ndim == 1 and len(bbox) == 6
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.type(torch.int32)
        binary_mask = torch.zeros(img_shape, dtype=torch.float32, device=bbox.device)
    else:
        bbox = bbox.astype(np.int32)
        binary_mask = np.zeros(img_shape, np.float32)
    binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1, bbox[4]:bbox[5] + 1] = 1
    return binary_mask
