import argparse
import csv
import datetime
import logging
import logging.handlers
import os
import sys
import threading

import colorlog
import numpy as np
import torch
import yaml


class KThread(threading.Thread):
    """A subclass of threading.Thread, with a kill()
    method.

    Come from:
    Kill a thread in Python:
    http://mail.python.org/pipermail/python-list/2004-May/260937.html
    """

    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.killed = False

    def start(self):
        """Start the thread."""
        self.__run_backup = self.run
        self.run = self.__run  # Force the Thread to install our trace.
        threading.Thread.start(self)

    def __run(self):
        """Hacked run function, which installs the
        trace."""
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, why, arg):
        if why == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, why, arg):
        if self.killed:
            if why == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True


class TimeoutException(Exception):
    """function run timeout"""


def timeout(seconds):
    """decorator for timeout"""

    def timeout_decorator(func):
        """true decorator"""

        def _new_func(oldfunc, result, oldfunc_args, oldfunc_kwargs):
            result.append(oldfunc(*oldfunc_args, **oldfunc_kwargs))

        def _(*args, **kwargs):
            result = []
            new_kwargs = {'oldfunc': func, 'result': result, 'oldfunc_args': args, 'oldfunc_kwargs': kwargs}
            thd = KThread(target=_new_func, args=(), kwargs=new_kwargs)
            thd.start()
            thd.join(seconds)
            alive = thd.is_alive()
            thd.kill()

            if alive:
                try:
                    raise TimeoutException(u'function run too long, timeout %d seconds.' % seconds)
                finally:
                    return u'function run too long, timeout %d seconds.' % seconds
            else:
                return result[0]

        _.__name__ = func.__name__
        _.__doc__ = func.__doc__
        return _

    return timeout_decorator


def load_config(config_filename) -> dict:
    with open(config_filename, 'r', encoding='utf8') as f:
        data = yaml.safe_load(f)
    return data


def save_config(config_filename, data):
    maybe_create_path(os.path.dirname(config_filename))
    with open(config_filename, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_dict_csv(filename: str, return_fieldnames=False) -> (list, list):
    with open(filename) as f:
        f_csv = csv.DictReader(f)
        data = list(f_csv)
        field_names = f_csv.fieldnames
    if return_fieldnames:
        return data, field_names
    else:
        return data


def write_dict_csv(filename: str, fieldnames: list, data: list):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow(item)


def load_thresholds(filename):
    assert os.path.exists(filename)
    thresholds = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip()
            thresholds.append(float(line))
    return thresholds


def save_thresholds(filename, thresholds):
    maybe_create_path(os.path.dirname(filename))
    with open(filename, 'w') as f:
        for th in thresholds:
            f.write('%s\n' % th)
    return


def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_logger(name, logging_folder=None, verbose=False, logging_prefix=None):
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logging.getLogger('PIL').setLevel(logging.INFO)  # prevent PIL logging many debug msgs
    logging.getLogger('matplotlib').setLevel(logging.INFO)  # prevent matplotlib logging many debug msgs

    # root logger to log everything
    root_logger = logging.root
    root_logger.setLevel(level)
    if not root_logger.handlers:
        format_str = '%(asctime)s [%(threadName)s] %(levelname)s [%(name)s] - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'cyan',
                  'INFO': 'green',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'red',
                  'CRITICAL': 'bold_red', }
        color_formatter = colorlog.ColoredFormatter(cformat, date_format, log_colors=colors)
        plain_formatter = logging.Formatter(format_str, date_format)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(color_formatter)
        root_logger.addHandler(stream_handler)
        # Logging to file
        if logging_folder is not None:
            maybe_create_path(logging_folder)
            logging_filename = datetime.datetime.now().strftime('%Y-%m-%d#%H-%M-%S') + '.log'
            if logging_prefix is not None:
                logging_filename = logging_prefix + '_' + logging_filename
            logging_filename = os.path.join(logging_folder, logging_filename)
            file_handler = logging.handlers.RotatingFileHandler(
                logging_filename, maxBytes=5 * 1024 * 1024, encoding='utf8')  # 5MB per file
            file_handler.setFormatter(plain_formatter)
            root_logger.addHandler(file_handler)
    return logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_devices(devices_arg: str, logger: logging.Logger):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    devices = devices_arg.replace(' ', '').split(',')
    if len(devices) > 1 and 'cpu' in devices:
        logger.warning('cannot run on both cpu and gpu. use gpu')
        devices.remove('cpu')
    if devices_arg == 'cpu' or not torch.cuda.is_available():
        logger.info('use cpu')
        return [torch.device('cpu')]

    cuda_count = torch.cuda.device_count()

    for dev in devices:
        if int(dev) >= cuda_count or int(dev) < 0:
            logger.warning('device %s is not available.' % dev)
            devices.remove(dev)
            continue
    if len(devices) == 0:
        logger.warning('no selected device is available, use cpu')
        return [torch.device('cpu')]
    return [torch.device('cuda', int(i)) for i in devices]


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self, init_sum=0.0, init_count=0.0):
        self.count = init_count
        self.sum = init_sum
        self.avg = 0.0 if self.count == 0.0 else self.sum / self.count

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def one_hot(arr, num_classes, axis=-1):
    assert isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)
    shape = list(arr.shape)
    if isinstance(arr, np.ndarray):
        arr = np.reshape(arr, -1)
        arr = np.eye(num_classes, dtype=arr.dtype)[arr]
        arr = np.reshape(arr, shape + [num_classes])
    else:
        arr = torch.reshape(arr, [-1])
        arr = torch.eye(num_classes, dtype=arr.dtype, device=arr.device)[arr]
        arr = torch.reshape(arr, shape + [num_classes])
    if axis != -1:
        arr = transpose(arr, axis, -1)
    return arr


def transpose(arr, first_index=1, second_index=-1):
    assert isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)
    assert arr.ndim > max(first_index, second_index)
    if isinstance(arr, np.ndarray):
        permute = list(range(arr.ndim))
        temp = permute[first_index]
        permute[first_index] = permute[second_index]
        permute[second_index] = temp
        return np.transpose(arr, permute)
    else:
        return torch.transpose(arr, first_index, second_index)


def transpose_move_to_end(arr, index=1):
    assert isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)
    assert arr.ndim > index
    permute = list(range(arr.ndim))
    permute.pop(index)
    permute.append(index)
    if isinstance(arr, np.ndarray):
        return np.transpose(arr, permute)
    else:
        return arr.permute(permute)
