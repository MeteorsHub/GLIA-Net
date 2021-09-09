import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from tensorboardX import SummaryWriter

from data_loader import get_instances_from_file_or_folder
from model.layers import SwitchNorm3d
from utils.losses import get_loss_fns, get_pyramid_weights
from utils.metrics import get_evaluation_metric
from utils.model_utils import create_optimizer, create_lr_scheduler, CheckpointLogger, bbox2binary_mask, get_model
from utils.project_utils import RunningAverage, maybe_create_path, write_dict_csv
from utils.visualization import get_tensorboard_formatter, get_text_image


class _ModelCore:
    def __init__(self,
                 config,
                 exp_path,
                 devices,
                 train_loader=None,
                 eval_loader=None,
                 test_loader=None,
                 logger=logging.getLogger('ModelCore')):
        self.logger = logger
        self.config = config
        self.exp_path = exp_path
        self.devices = devices
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        self.checkpoint_dir = os.path.join(exp_path, config['ckpt_folder'])
        self.summary_dir = os.path.join(exp_path, config['summary_folder'])
        self.batch_size = config['train']['batch_size'] * len(devices)
        self.num_iterations = 1
        self.num_epoch = 1
        self.best_eval_score = None

        self.model = get_model(config, logger)
        self.logger.debug(self.model)
        self.optimizer = None  # create in sub-classes
        if len(devices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=devices, output_device=devices[0])
        self.model.to(devices[0])
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        if 'visualization' in config:
            self.tensorboard_image_formatter = \
                get_tensorboard_formatter(config['visualization']['name'], **config['visualization'])
        else:
            self.tensorboard_image_formatter = None

        self.checkpoint_logger = \
            CheckpointLogger(self.checkpoint_dir, logger, config['train'].get('max_keep_runs', None))
        self.load_best = config.get('load_best_model', False)

    def _wrap_inputs(self, inputs):
        if self.config['task'] == 'AneurysmSeg':
            if self.config['model']['with_global']:
                return inputs['local_cta_input'], inputs['global_cta_input'], inputs['global_patch_location_bbox']
            else:
                return inputs['local_cta_input']
        else:
            self.logger.critical('Cannot recognize task: %s' % self.config['task'])
            exit(1)

    def _wrap_outputs(self, outputs):
        if self.config['task'] == 'AneurysmSeg':
            outputs_dict = OrderedDict()
            if self.config['model']['with_global']:
                outputs_dict['local_output'] = outputs[0]
                outputs_dict['global_output'] = outputs[1]
                return outputs_dict
            else:
                outputs_dict['local_output'] = outputs
                return outputs_dict
        else:
            self.logger.critical('Cannot recognize task: %s' % self.config['task'])
            exit(1)

    def _split_and_place_batch(self, data):
        def _move_to_device(inputs):
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                return tuple([_move_to_device(x) for x in inputs])
            elif isinstance(inputs, dict) or isinstance(inputs, OrderedDict):
                return dict(zip(list(inputs.keys()), [_move_to_device(x) for x in inputs.values()]))
            elif isinstance(inputs, str):
                return inputs
            elif inputs is None:
                return None
            else:
                return inputs.to(self.devices[0])

        placed_data = _move_to_device(data)
        return placed_data

    def _build_batch_and_normalize(self, inputs, targets=None):
        # all as dict
        model_inputs = OrderedDict()  # the first is main
        if targets is not None:
            model_targets = OrderedDict()  # the first is main
            model_weights = OrderedDict()  # the first is main
        raw_inputs = OrderedDict()  # for visualization
        hu_intervals = self.config['data']['hu_values']

        def _normalize(image):
            normalized_img = []
            for hu_inter in hu_intervals:
                hu_channel = torch.clamp(image, hu_inter[0], hu_inter[1])
                # norm to 0-1
                normalized_img.append((hu_channel - hu_inter[0]) / (hu_inter[1] - hu_inter[0]))
            normalized_img = torch.stack(normalized_img, dim=1)
            return normalized_img

        if self.config['task'] == 'AneurysmSeg':
            model_inputs['local_cta_input'] = _normalize(inputs['cta_img']).type(torch.float32)
            raw_inputs['local_cta_input'] = torch.unsqueeze(inputs['cta_img'].type(torch.float32), 1)
            if targets is not None:
                model_targets['local_ane_seg_target'] = targets['aneurysm_seg'].type(torch.int64)
                local_weight_config = self.config['train']['losses'][0]['weight']
                # compute local_ane_seg_target_weight. only for computed weight type like pyramid
                if local_weight_config['type'] == 'pyramid':
                    model_weights['local_ane_seg_target_weight'] = \
                        get_pyramid_weights(model_targets['local_ane_seg_target'], **local_weight_config)
                else:
                    model_weights['local_ane_seg_target_weight'] = None
            # with global positioning network
            if self.config['model'].get('with_global', False):
                model_inputs['global_cta_input'] = _normalize(inputs['global_cta_img']).type(torch.float32)
                raw_inputs['global_cta_input'] = torch.unsqueeze(inputs['global_cta_img'].type(torch.float32), 1)
                model_inputs['global_patch_location_bbox'] = inputs['global_patch_location_bbox'].type(torch.float32)
                raw_inputs['global_patch_location_bbox'] = inputs['global_patch_location_bbox'].type(torch.float32)
                if targets is not None:
                    model_targets['global_ane_cls_target'] = targets['global_aneurysm_label'].type(torch.int64)
                    model_weights['global_ane_cls_target_weight'] = None  # pyramid not used
        else:
            self.logger.critical('Cannot recognize task: %s' % self.config['task'])
            exit(1)

        if targets is not None:
            for v in model_weights.values():
                if v is not None:
                    v.to(self.devices[0])
            return model_inputs, model_targets, model_weights, raw_inputs
        else:
            return model_inputs, raw_inputs

    def _compute_loss_and_output(self, raw_outputs, targets=None, weights=None):
        if targets is not None:
            losses = OrderedDict()
            if self.config['task'] == 'AneurysmSeg':
                losses['local_loss'] = self.config['train']['losses'][0]['loss_weight'] * self.loss_fns[0](
                    raw_outputs['local_output'], targets['local_ane_seg_target'], weights['local_ane_seg_target_weight']
                )
                if self.config['model']['with_global']:
                    losses['global_loss'] = self.config['train']['losses'][1]['loss_weight'] * self.loss_fns[1](
                        raw_outputs['global_output'], targets['global_ane_cls_target'],
                        weights['global_ane_cls_target_weight']
                    )
                    losses['total_loss'] = losses['local_loss'] + losses['global_loss']
                else:
                    losses['local_loss'] = self.config['train']['losses'][0]['loss_weight'] * self.loss_fns[0](
                        raw_outputs['local_output'], targets['local_ane_seg_target'],
                        weights['local_ane_seg_target_weight']
                    )
                    losses['total_loss'] = losses['local_loss']
                losses.move_to_end('total_loss', last=False)
            else:
                self.logger.critical('Cannot recognize task: %s' % self.config['task'])
                exit(1)

        # apply final activation to outputs
        outputs = OrderedDict()
        for j, key in enumerate(raw_outputs.keys()):
            if self.config['train']['losses'][j]['final_activation'] == 'softmax':
                outputs[key] = torch.nn.Softmax(dim=1)(raw_outputs[key])
            elif self.config['train']['losses'][j]['final_activation'] == 'sigmoid':
                fg_output = torch.nn.Sigmoid()(raw_outputs[key])
                outputs[key] = torch.cat([1 - fg_output, fg_output], dim=1)
            elif self.config['train']['losses'][j]['final_activation'] == 'identity':
                outputs[key] = raw_outputs[key]
        # outputs and losses have same length

        if targets is None:
            return outputs
        else:
            return outputs, losses

    def _save_checkpoint(self, is_best, epoch_finished=False):
        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        state_dict = {
            'num_epoch': self.num_epoch,
            'epoch_finished': epoch_finished,
            'num_iteration': self.num_iterations,
            'model_state_dict': model_state_dict
        }
        if self.best_eval_score is not None:
            state_dict['best_eval_score'] = self.best_eval_score
        if self.optimizer is not None:
            state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        self.checkpoint_logger.save_checkpoint(state_dict, is_best)

    def _load_checkpoint(self, ckpt_file=None, is_best=False):
        if self.checkpoint_logger.is_ckpt_exist(self.load_best):
            if isinstance(self.model, torch.nn.DataParallel):
                model = self.model.module
            else:
                model = self.model
            if ckpt_file is None:
                state = self.checkpoint_logger.load_ckeckpoint(model, is_best=is_best, optimizer=self.optimizer)
            else:
                state = self.checkpoint_logger.load_ckeckpoint(model, ckpt_file=ckpt_file, optimizer=self.optimizer)
            if state['epoch_finished']:
                self.num_epoch = state['num_epoch'] + 1
            else:
                self.num_epoch = state['num_epoch']
            self.num_iterations = state['num_iteration'] + 1
            if 'best_eval_score' in state:
                if isinstance(state['best_eval_score'], torch.Tensor):
                    self.best_eval_score = state['best_eval_score'].item()
                else:
                    self.best_eval_score = state['best_eval_score']
        else:
            self.logger.warning('No checkpoint exists. Starting from scratch...')

    def _log_lr(self, optimizer=None, summary_steps=None):
        if optimizer is None:
            self.logger.warning('log_lr() needs valid optimizer which is None now.')
            return
        if summary_steps is None:
            summary_steps = self.num_iterations
        lr = optimizer.param_groups[0]['lr']
        self.writer.add_scalar('general/learning_rate', lr, summary_steps)

    def _log_stats(self, phase, losses: OrderedDict, metrics=None, is_avg=True, summary_steps=None, step_time=None):
        if summary_steps is None:
            summary_steps = self.num_iterations
        tag_value = OrderedDict()
        if is_avg:
            prefix = 'epoch_wise'
        else:
            prefix = 'step_wise'
        if step_time is not None:
            self.writer.add_scalar(f'general/{phase}_step_time', step_time, summary_steps)
        for loss_name, loss_value in losses.items():
            tag_value[f'{phase}_loss_{prefix}/{loss_name}'] = loss_value.avg if is_avg else loss_value.item()
        if metrics is not None:
            for metric_name, metric_fn in metrics.items():
                tag_value[f'{phase}_metric_{prefix}/{metric_name}'] = metric_fn.result if is_avg else metric_fn.item()
                if is_avg:
                    # log metrics to file
                    if hasattr(metric_fn, 'data_dict') and self.config['eval'].get('save_metrics_to_file',
                                                                                   None) is not None:
                        field_names, data_dict = metric_fn.data_dict
                        filename = os.path.join(self.exp_path,
                                                metric_name + '_' + str(summary_steps) + '_' + self.config['eval'][
                                                    'save_metrics_to_file'])
                        write_dict_csv(filename, field_names, data_dict)

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, summary_steps)

    def _log_params(self, summary_steps=None):
        if summary_steps is None:
            summary_steps = self.num_iterations
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), summary_steps)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), summary_steps)
        for name, value in self.model.named_modules():
            if isinstance(value, torch.nn.BatchNorm3d) or isinstance(value, SwitchNorm3d):
                self.writer.add_histogram(name + '.running_mean', value.running_mean.data.cpu().numpy(), summary_steps)
                self.writer.add_histogram(name + '.running_var', value.running_var.data.cpu().numpy(), summary_steps)

    def _log_images(self, phase, inputs=OrderedDict(), targets=OrderedDict(), outputs=OrderedDict(), weights=None,
                    metas=None,
                    summary_steps=None):
        if len(inputs) == 0 and len(targets) == 0 and len(outputs) == 0:
            self.logger.warning('no imgs to log because inputs, targets and outputs are all None')
        # general preprocess
        if summary_steps is None:
            summary_steps = self.num_iterations
        for k in inputs.keys():
            inputs[k] = inputs[k].detach().cpu().numpy().astype(np.float32)  # tensor to ndarray
            if 'mask' not in k and 'bbox' not in k:  # normalize cta image to 0-1
                hu_low = self.config['data']['hu_values'][0][0]
                hu_high = self.config['data']['hu_values'][-1][1]
                inputs[k] = np.clip(inputs[k], hu_low, hu_high)
                inputs[k] = (inputs[k] - hu_low) / (hu_high - hu_low)
            if inputs[k].ndim == 4:
                inputs[k] = np.expand_dims(inputs[k], 1)
        for k in outputs.keys():
            outputs[k] = outputs[k].detach().cpu().numpy().astype(np.float32)
            if outputs[k].ndim == 4:
                outputs[k] = np.expand_dims(outputs[k], 1)
        for k in targets.keys():
            targets[k] = targets[k].detach().cpu().numpy().astype(np.float32)
            if targets[k].ndim == 4:
                targets[k] = np.expand_dims(targets[k], 1)
        for k in weights.keys():
            if weights[k] is None:
                pass
            else:
                weights[k] = weights[k].detach().cpu().numpy().astype(np.float32)
                if weights[k].ndim == 4:
                    weights[k] = np.expand_dims(weights[k], 1)
                weights[k] = np.clip(weights[k], 0, 30) / 30.0  # normalize

        # task specific process
        if self.config['task'] == 'AneurysmSeg':
            for k in outputs.keys():
                if outputs[k].shape[1] == 2:
                    outputs[k] = outputs[k][:, 1:2, ...]  # 2 classes softmax output
            if 'global_output' in outputs:
                text_images = np.stack([get_text_image(text='%1.4f' % outputs['global_output'][i][0],
                                                       image_h=inputs['global_cta_input'].shape[3],
                                                       image_w=inputs['global_cta_input'].shape[4])
                                        for i in range(len(outputs['global_output']))])
                text_images = np.transpose(np.expand_dims(text_images, 1), (0, 4, 1, 2, 3))
                text_images = np.tile(text_images, (1, 1, inputs['global_cta_input'].shape[2], 1, 1))
                outputs['global_output'] = text_images
            if 'global_ane_cls_target' in targets:
                texts = ['' for _ in range(self.batch_size)]
                if 'id' in metas:
                    texts = [texts[i] + metas['id'][i] + '\n' for i in range(self.batch_size)]
                if 'hospital' in metas:
                    texts = [texts[i] + metas['hospital'][i] + '\n' for i in range(self.batch_size)]
                texts = [texts[i] + '%1.4f' % targets['global_ane_cls_target'][i] for i in range(self.batch_size)]
                text_images = np.stack([get_text_image(text=texts[i],
                                                       image_h=inputs['global_cta_input'].shape[3],
                                                       image_w=inputs['global_cta_input'].shape[4])
                                        for i in range(self.batch_size)])
                text_images = np.transpose(np.expand_dims(text_images, 1), (0, 4, 1, 2, 3))
                text_images = np.tile(text_images, (1, 1, inputs['global_cta_input'].shape[2], 1, 1))
                targets['global_ane_cls_target'] = text_images
            if 'global_patch_location_bbox' in inputs:
                # expand to RGB
                inputs['global_cta_input'] = np.tile(inputs['global_cta_input'], (1, 3, 1, 1, 1))
                # patch_location_mask to red
                global_path_location_mask = np.expand_dims(bbox2binary_mask(inputs['global_patch_location_bbox'],
                                                                            inputs['global_cta_input'].shape[2:]), 1)

                alpha = 0.3 * global_path_location_mask[:, 0]
                inputs['global_cta_input'][:, 0] = inputs['global_cta_input'][:, 0] * (1 - alpha) + \
                                                   global_path_location_mask[:, 0] * alpha
                inputs.pop('global_patch_location_bbox')
            if 'local_ane_seg_target_weight' in weights:
                if weights['local_ane_seg_target_weight'] is None:
                    weights.pop('local_ane_seg_target_weight')
            if 'global_ane_cls_target_weight' in weights:
                weights.pop('global_ane_cls_target_weight')

        def _log_img(_imgs_dict):
            _tag, _image = self.tensorboard_image_formatter(_imgs_dict, max_num_samples=4)
            _tag = phase + '/' + _tag
            if _image is not None:
                self.writer.add_image(_tag, _image, summary_steps, dataformats='CHW')

        # related input, target and output
        max_len = max(len(inputs), len(targets), len(outputs))
        for i in range(max_len):
            imgs_dict = OrderedDict()
            if i < len(inputs):
                imgs_dict[list(inputs.keys())[i]] = list(inputs.values())[i]
            if i < len(targets):
                imgs_dict[list(targets.keys())[i]] = list(targets.values())[i]
            if i < len(outputs):
                imgs_dict[list(outputs.keys())[i]] = list(outputs.values())[i]
            if i < len(weights):
                imgs_dict[list(weights.keys())[i]] = list(weights.values())[i]
            _log_img(imgs_dict)

    def _log_eval_curves(self, phase, eval_curve_fns=None):
        # should be called after each epoch
        if eval_curve_fns is None:
            self.logger.warning('log_eval_curves() needs valid eval_curve_fns.')
            return
        for k, v in eval_curve_fns.items():
            img = v.curve
            self.writer.add_image(f"{phase}/{k}", img, self.num_iterations, dataformats='HWC')


class Trainer(_ModelCore):
    def __init__(self,
                 config,
                 exp_path,
                 devices,
                 train_loader=None,
                 eval_loader=None,
                 logger=logging.getLogger('Trainer')):
        super(Trainer, self).__init__(config, exp_path, devices, train_loader, eval_loader, logger=logger)

        self.log_every_n_iters = config['log_every_n_iters']
        self.max_num_epochs = config['train'].get('num_epoch', 10)
        self.eval_every_n_epochs = config['train'].get('eval_every_n_epochs', 1)
        self.log_summary_every_n_iters = config['train'].get('log_summary_every_n_iters', 1)
        self.save_ckpt_every_n_iters = config['train'].get('save_ckpt_every_n_iters', 10)
        self.eval_score_higher_is_better = config['eval'].get('eval_score_higher_is_better', True)

        self.loss_fns = get_loss_fns(config, device=devices[0], logger=logger)
        # the first will be used as main eval metric
        self.eval_metric_fns, self.eval_curve_fns = get_evaluation_metric(config, logger, devices[0])

        if self.eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        self.has_log_graph = False
        self.skip_eval_metric_in_training = config['eval'].get('skip_eval_metric_in_training', False)
        self.optimizer = create_optimizer(config, self.model)
        self.lr_scheduler = create_lr_scheduler(config, self.optimizer)

        if config.get('ckpt_file', None) is None:
            self._load_checkpoint(is_best=self.load_best)
        else:
            self._load_checkpoint(ckpt_file=config['ckpt_file'])

    def train(self):
        for _ in range(self.num_epoch, self.max_num_epochs + 1):
            # train for one epoch
            self.train_epoch()
            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.StepLR):
                    self.lr_scheduler.step(self.num_epoch - 1)
            # occasionally eval for one epoch
            if self.eval_every_n_epochs != 0 and self.num_epoch % self.eval_every_n_epochs == 0:
                # evaluate on evaluation set
                eval_score = self.evaluate_epoch()
                # adjust learning rate if necessary
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(eval_score)
                self._log_lr(self.optimizer)
                is_best = self._is_best_eval_score(eval_score)
                self._save_checkpoint(is_best, epoch_finished=True)
            else:
                self._save_checkpoint(is_best=False, epoch_finished=True)

            self.num_epoch += 1
        self.logger.info('All the %d epochs are finished' % (self.num_epoch - 1))

    def train_epoch(self, phase='train'):
        if self.train_loader is None:
            self.logger.error('Try to %s but there is no train_loader' % phase)
            return 0.0
        avg_losses = None
        # sets the model in training mode
        self.model.train()
        self._reset_eval_metrics()
        epoch_start_time = time.time()
        self.logger.info('Begin to %s on epoch %d/%d...' % (phase, self.num_epoch, self.max_num_epochs))

        since = time.time()
        time_per_iter = None
        for i, data in enumerate(self.train_loader):
            raw_inputs, targets, metas = self._split_and_place_batch(data)
            inputs, targets, weights, raw_inputs = self._build_batch_and_normalize(raw_inputs, targets)

            raw_outputs = self._wrap_outputs(self.model(self._wrap_inputs(inputs)))
            outputs, losses = self._compute_loss_and_output(raw_outputs, targets, weights)

            main_output = list(outputs.values())[0]
            main_target = list(targets.values())[0]
            main_loss = list(losses.values())[0]

            # update avg_losses
            if avg_losses is None:
                avg_losses = OrderedDict(zip(list(losses.keys()), [RunningAverage() for _ in range(len(losses))]))
            for loss_key, loss_value in losses.items():
                avg_losses[loss_key].update(loss_value.item(), self.batch_size)

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.StepLR):
                self.lr_scheduler.step(self.num_iterations)

            # compute metrics
            if not self.skip_eval_metric_in_training:
                if (i + 1) % self.log_every_n_iters == 0 or self.num_iterations % self.log_summary_every_n_iters == 0:
                    current_metrics = OrderedDict()
                    for key, metric_fn in self.eval_metric_fns.items():
                        current_metrics[key] = metric_fn(main_output, main_target, **metas)

            # log to logger
            if (i + 1) % self.log_every_n_iters == 0:
                time_per_iter = (time.time() - since) / self.log_every_n_iters
                logging_info = '(Time per iter: %1.2fs)%s iter %d. epoch %d/%d.' \
                               % (time_per_iter, phase, self.num_iterations, self.num_epoch, self.max_num_epochs)
                for loss_name, loss_value in losses.items():
                    logging_info += ' %s: %1.4f' % (loss_name, loss_value.item())
                if not self.skip_eval_metric_in_training:
                    for metric_name, metric_value in current_metrics.items():
                        if isinstance(metric_value.item(), int):
                            logging_info += ' %s: %d' % (metric_name, metric_value.item())
                        else:
                            logging_info += ' %s: %1.4f' % (metric_name, metric_value.item())
                self.logger.info(logging_info)
                since = time.time()

            # log to summary folder
            if self.num_iterations % self.log_summary_every_n_iters == 0:
                if not self.skip_eval_metric_in_training:
                    self._log_stats(phase, losses=losses, metrics=current_metrics,
                                    is_avg=False, summary_steps=self.num_iterations, step_time=time_per_iter)
                else:
                    self._log_stats(phase, losses=losses, is_avg=False, step_time=time_per_iter)
                self._log_lr(self.optimizer)
                self._log_images('train', raw_inputs, targets, outputs, weights, metas)

            # save ckpt
            if self.num_iterations % self.save_ckpt_every_n_iters == 0:
                self._save_checkpoint(is_best=False, epoch_finished=False)

            self.num_iterations += 1
        if avg_losses is None:
            self.logger.error('No %s data fetched. Try to restart...' % phase)
            return self.train_epoch(phase)
        # epoch log to summary folder
        self._log_stats(phase, losses=avg_losses, metrics=self.eval_metric_fns,
                        is_avg=True, summary_steps=self.num_iterations)
        if not self.skip_eval_metric_in_training:
            self._log_eval_curves(phase, self.eval_curve_fns)
        # epoch log to logger
        logging_info = '(Time epoch: %1.2f)%s epoch %d/%d finished.' \
                       % (time.time() - epoch_start_time, phase, self.num_epoch, self.max_num_epochs)
        for loss_name, loss_value in avg_losses.items():
            logging_info += ' %s_avg: %1.4f' % (loss_name, loss_value.avg)
        if not self.skip_eval_metric_in_training:
            for metric_name, metric_fn in self.eval_metric_fns.items():
                if isinstance(metric_fn.result.item(), int):
                    logging_info += ' %s: %d' % (metric_name, metric_fn.result.item())
                else:
                    logging_info += ' %s: %1.4f' % (metric_name, metric_fn.result.item())
        self.logger.info(logging_info)

    def evaluate_epoch(self, phase='eval'):
        if self.eval_loader is None:
            self.logger.error('Try to %s but there is no eval_loader' % phase)
            return 0.0
        self.logger.info('%s epoch %d...' % (phase, self.num_epoch))
        eval_avg_losses = None
        self._reset_eval_metrics()
        self.model.eval()

        epoch_start_time = time.time()
        eval_summary_steps = self.num_iterations  # in order to log all eval images
        time_per_iter = None

        with torch.no_grad():
            since = time.time()
            for i, data in enumerate(self.eval_loader):
                raw_inputs, targets, metas = self._split_and_place_batch(data)
                inputs, targets, weights, raw_inputs = self._build_batch_and_normalize(raw_inputs, targets)

                raw_outputs = self._wrap_outputs(self.model(self._wrap_inputs(inputs)))
                outputs, losses = self._compute_loss_and_output(raw_outputs, targets, weights)

                main_output = list(outputs.values())[0]
                main_target = list(targets.values())[0]

                if eval_avg_losses is None:
                    eval_avg_losses = OrderedDict(zip(list(losses.keys()), [RunningAverage() for _ in range(len(losses))]))
                for loss_key, loss_value in losses.items():
                    eval_avg_losses[loss_key].update(loss_value.item(), self.batch_size)

                current_metrics = OrderedDict()
                for key, metric_fn in self.eval_metric_fns.items():
                    current_metrics[key] = metric_fn(main_output, main_target, **metas)

                # log to logger
                if (i + 1) % self.log_every_n_iters == 0:
                    time_per_iter = (time.time() - since) / self.log_every_n_iters
                    logging_info = '(Time per iter: %1.2fs)%s iter %d. epoch %d/%d.' \
                                   % (time_per_iter, phase, i + 1, self.num_epoch, self.max_num_epochs)
                    for loss_name, loss_value in losses.items():
                        logging_info += ' %s: %1.4f' % (loss_name, loss_value.item())
                    for metric_name, metric_value in current_metrics.items():
                        if isinstance(metric_value.item(), int):
                            logging_info += ' %s: %d' % (metric_name, metric_value.item())
                        else:
                            logging_info += ' %s: %1.4f' % (metric_name, metric_value.item())
                    self.logger.info(logging_info)
                    since = time.time()

                if eval_summary_steps % ((self.log_summary_every_n_iters // 5) + 1) == 0:
                    # log to summary
                    self._log_stats(phase, losses=losses, metrics=current_metrics,
                                    is_avg=False, summary_steps=eval_summary_steps, step_time=time_per_iter)
                    self._log_images(phase, raw_inputs, targets, outputs, weights, metas,
                                     summary_steps=eval_summary_steps)
                eval_summary_steps += 1

            if eval_avg_losses is None:
                self.logger.error('No %s data fetched. Try to restart...' % phase)
                return self.evaluate_epoch(phase)
            # epoch log to summary folder
            self._log_stats(phase, losses=eval_avg_losses, metrics=self.eval_metric_fns,
                            is_avg=True, summary_steps=self.num_iterations)
            self._log_eval_curves(phase, self.eval_curve_fns)
            # epoch log to logger
            logging_info = '(Time epoch: %1.2f)%s finished.' % (time.time() - epoch_start_time, phase)
            for loss_name, loss_value in eval_avg_losses.items():
                logging_info += ' %s_avg: %1.4f' % (loss_name, loss_value.avg)
            for metric_name, metric_fn in self.eval_metric_fns.items():
                if isinstance(metric_fn.result.item(), int):
                    logging_info += ' %s: %d' % (metric_name, metric_fn.result.item())
                else:
                    logging_info += ' %s: %1.4f' % (metric_name, metric_fn.result.item())
            self.logger.info(logging_info)
            main_eval_score = list(self.eval_metric_fns.values())[0].result
            return main_eval_score

    def _reset_eval_metrics(self):
        for metric_fn in self.eval_metric_fns.values():
            metric_fn.reset()

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score
        if is_best:
            self.logger.info(f'Got new best evaluation score: {eval_score}')
            self.best_eval_score = eval_score.item()
        return is_best


class Evaler(Trainer):
    def __init__(self,
                 config,
                 exp_path,
                 devices,
                 eval_loader=None,
                 logger=logging.getLogger('Evaler')):
        super(Evaler, self).__init__(config, exp_path, devices, eval_loader=eval_loader, logger=logger)
        self.eval_phase = config['eval'].get('phase', 'test')

        if config.get('ckpt_file', None) is None:
            self._load_checkpoint(is_best=self.load_best)
        else:
            self._load_checkpoint(ckpt_file=config['ckpt_file'])

    def eval(self):
        # eval for one epoch
        eval_score = self.evaluate_epoch(self.eval_phase)
        is_best = self._is_best_eval_score(eval_score)
        self.logger.info('Eval finished. Get %seval score %1.4f' % ('best ' if is_best else '', eval_score))


class Inferencer(_ModelCore):
    def __init__(self,
                 config,
                 exp_path,
                 devices,
                 inference_file_or_folder,
                 output_folder=None,
                 input_type='nii',
                 save_binary=True,
                 save_prob=False,
                 save_global=False,
                 test_loader_manager=None,
                 logger=logging.getLogger('Inferencer')):
        super(Inferencer, self).__init__(config, exp_path, devices, test_loader=test_loader_manager.test_loader,
                                         logger=logger)
        self.test_loader_manager = test_loader_manager
        self.test_phase = config['eval'].get('phase', 'inference')
        if input_type not in ['nii', 'dcm']:
            self.logger.critical('input_type must be nii or dcm')
            exit(1)
        if not os.path.exists(inference_file_or_folder):
            self.logger.critical('inference_file_or_folder %s does not exist.' % inference_file_or_folder)
            exit(1)

        self.inference_file_or_folder = inference_file_or_folder
        self.output_folder = output_folder
        self.input_type = input_type
        self.save_binary = save_binary
        self.save_prob = save_prob
        self.save_global = save_global
        if 'eval' in config:
            self.prob_threshold = config['eval'].get('probability_threshold', 0.5)
        else:
            self.prob_threshold = 0.5

        if config.get('ckpt_file', None) is None:
            self._load_checkpoint(is_best=self.load_best)
        else:
            self._load_checkpoint(ckpt_file=config['ckpt_file'])

    def inference(self):
        if self.test_loader is None:
            self.logger.error('Try to %s but there is no test_loader' % self.test_phase)
            return 0.0

        self.logger.info('Begin to scan input_folder_or_file %s...' % self.inference_file_or_folder)
        instances = get_instances_from_file_or_folder(self.inference_file_or_folder, instance_type=self.input_type)

        for i, instance in enumerate(instances):
            if self.output_folder is None:
                if self.input_type == 'nii':
                    output_file = instance.replace('.nii.gz', '_pred.nii.gz')
                elif self.input_type == 'dcm':
                    output_file = os.path.join(os.path.dirname(instance[0]), 'prediction.nii.gz')
            else:
                assert not instance[0].startswith(self.output_folder)  # in case override original
                if self.input_type == 'nii':
                    output_file = os.path.join(self.output_folder, os.path.basename(instance))
                elif self.input_type == 'dcm':
                    output_file = os.path.join(self.output_folder,
                                               os.path.basename(os.path.dirname(instance[0])) + '.nii.gz')
            self.inference_instance(instance, output_file)
            self.logger.info('finish %d in %d instances' % (i + 1, len(instances)))

    def inference_instance(self, input_file_s, output_file):
        self.test_loader_manager.load(input_file_s, input_type=self.input_type)
        self.model.eval()

        instance_start_time = time.time()
        prediction_instance_shape = self.test_loader_manager.instance_shape
        prediction_patch_starts = self.test_loader_manager.patch_starts
        prediction_patch_size = self.test_loader_manager.patch_size

        prediction = np.zeros(prediction_instance_shape.tolist(), dtype=np.float32)
        overlap_count = np.zeros(prediction_instance_shape.tolist(), dtype=np.float32)
        if self.save_global:
            global_map = np.zeros(prediction_instance_shape.tolist(), dtype=np.float32)

        if isinstance(input_file_s, list) or isinstance(input_file_s, tuple):
            input_instance = os.path.dirname(input_file_s[0])
        else:
            input_instance = input_file_s

        preprocess_time = time.time() - instance_start_time

        self.logger.info('%s instance %s (%d patches)...'
                         % (self.test_phase, input_instance, len(prediction_patch_starts)))
        time_all_iters = 0
        time_loading = 0

        print('\r\tprocessing procedure: %1.1f%%' % 0, end='')
        with torch.no_grad():
            loading_since = time.time()
            for i, data in enumerate(self.test_loader):
                time_loading += time.time() - loading_since
                since = time.time()

                raw_inputs, metas = self._split_and_place_batch(data)
                inputs, raw_inputs = self._build_batch_and_normalize(raw_inputs)
                raw_outputs = self._wrap_outputs(self.model(self._wrap_inputs(inputs)))
                outputs = self._compute_loss_and_output(raw_outputs)
                main_output = list(outputs.values())[0][:, 1]  # the 2nd channel of output
                main_output = main_output.detach().cpu().numpy()
                if self.save_global:
                    global_output = list(outputs.values())[1][:, 1]
                    global_output = global_output.detach().cpu().numpy()

                # iter along batch
                for j, main_out in enumerate(main_output):
                    patch_starts = metas['patch_starts'][j]
                    patch_ends = [patch_starts[i] + prediction_patch_size[i] for i in range(3)]
                    prediction[patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
                    patch_starts[2]:patch_ends[2]] += main_out
                    overlap_count[patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
                    patch_starts[2]:patch_ends[2]] += 1
                    if self.save_global:
                        global_map[patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
                        patch_starts[2]:patch_ends[2]] += global_output[j]

                time_all_iters += time.time() - since
                print('\r\tprocessing procedure: %1.1f%%' % (
                            100 * self.batch_size * (i + 1) / len(prediction_patch_starts)), end='')
                loading_since = time.time()

            print('')
            if time_all_iters == 0:
                self.logger.error('\tNo %s data fetched from instance %s. Try to restart...'
                                  % (self.test_phase, input_instance))
                return self.inference_instance(input_file_s, output_file)

            summarize_since = time.time()
            print('\tsummarize and save...')
            overlap_count = np.where(overlap_count == 0, np.ones_like(overlap_count), overlap_count)
            prediction = prediction / overlap_count
            prediction = self.test_loader_manager.restore_spacing(prediction, is_mask=False)
            if self.save_global:
                global_map = global_map / overlap_count
                global_map = self.test_loader_manager.restore_spacing(global_map, is_mask=False)

            summarize_time = time.time() - summarize_since
            processing_time = time.time() - instance_start_time

            if self.output_folder is None:
                assert '_pred' in output_file  # in case override original
            else:
                maybe_create_path(self.output_folder)
            if self.save_binary:
                binary_prediction = (prediction > self.prob_threshold).astype(np.int32)
                self.test_loader_manager.save_prediction(binary_prediction, output_file)
            if self.save_prob:
                self.test_loader_manager.save_prediction(prediction, output_file.replace('.nii.gz', '_prob.nii.gz'))
            if self.save_global:
                self.test_loader_manager.save_prediction(global_map, output_file.replace('.nii.gz', '_global.nii.gz'))

            logging_info = '\t((total time: %1.2f; preprocess_time: %1.2f; loading_time: %1.2f; network time: %1.2f; summarize_time: %1.2f) Prediction of instance %s saved to %s. ' \
                           % (
                               processing_time, preprocess_time, time_loading, time_all_iters, summarize_time,
                               input_instance, output_file)
            print(logging_info)
            return prediction
