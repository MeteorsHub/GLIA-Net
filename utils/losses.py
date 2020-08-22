import logging

import numpy as np
import torch
import torch.nn as nn

from utils.project_utils import transpose_move_to_end

SUPPORTED_LOSSES = ['SoftmaxCrossEntropyLoss', 'FocalLoss', 'DiceLoss', 'ExpLoss', 'SmoothL1Loss', 'L1Loss']


class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones((class_num, 1))
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                if class_num == 2 and not isinstance(alpha, list):
                    self.alpha = torch.tensor([1 - alpha, alpha])
        self.gamma = gamma
        self.class_num = class_num

    def forward(self, inputs, targets):
        C = inputs.size(1)
        inputs = transpose_move_to_end(inputs, 1).contiguous().view(-1, C)
        P = torch.nn.functional.softmax(inputs, dim=1)
        N = inputs.size(0)

        class_mask = inputs.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)

        alpha = self.alpha[ids.view(-1)].view(-1, 1).to(inputs.device)

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        return loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, class_weight=None, smooth=1):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if class_weight is None:
            self.class_weight = [1.0 / num_classes] * num_classes
        else:
            assert len(class_weight) == num_classes
            self.class_weight = class_weight / class_weight.sum()

    def forward(self, inputs, targets):
        preds = torch.nn.functional.sigmoid(inputs)
        loss = inputs.new_zeros([])
        pred = preds[:, 0]
        target = targets
        intersection = (pred * target).sum()
        loss += (1 - (2. * intersection + self.smooth) / ((pred * pred).sum() + (target * target).sum() + self.smooth))
        return loss


class ExpLoss(nn.Module):
    def __init__(self, num_classes=2,
                 omg_dice=0.8, omg_cross=0.2, gamma_dice=0.3, gamma_cross=0.3, smooth=1.0, class_weight=None, **kwargs):
        super(ExpLoss, self).__init__()
        self.num_classes = num_classes
        self.omg_dice = omg_dice
        self.omg_cross = omg_cross
        self.gamma_dice = gamma_dice
        self.gamma_cross = gamma_cross
        self.smooth = smooth

        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weight, ignore_index=-100, reduction='none')

    def forward(self, inputs, targets, weight=None):
        y_pred = torch.sigmoid(inputs[:, 1])
        y_true = targets.type(torch.float32)

        inter = y_pred * y_true
        dice = (2 * (inter.sum(dim=(1, 2, 3))) + self.smooth) / ((y_true + y_pred).sum(dim=(1, 2, 3)) + self.smooth)
        loss_dice = ((- dice.log()).clamp_min_(1e-8) ** self.gamma_dice)

        loss_cross = (self.cross_entropy(inputs, targets).clamp_min_(1e-8)) ** self.gamma_cross
        if weight is not None:
            weight = weight.expand_as(loss_cross)
            loss_cross *= weight
        loss_cross = loss_cross.mean(dim=(1, 2, 3))
        loss = self.omg_dice * loss_dice + self.omg_cross * loss_cross
        return loss


class LossWrapper(nn.Module):
    def __init__(self, name, num_classes, logger, device, reduction='mean', ignored_index=None, weight=None, **kwargs):
        super(LossWrapper, self).__init__()
        self.loss_name = name
        self.weight_type = weight['type'] if weight is not None else None
        self.reduction = reduction
        self.ignored_index = ignored_index
        self.num_classes = num_classes
        self.logger = logger

        if self.weight_type == 'class':
            class_weight = torch.tensor(np.array(weight['class_weight_list'], np.float32))
            class_weight = class_weight.to(device)
        else:
            class_weight = None

        if self.loss_name == 'BCEWithLogitsLoss':
            if ignored_index is not None:
                if class_weight is None:
                    class_weight = np.ones([self.num_classes], np.float32)
                    class_weight[ignored_index] = 0
                    class_weight = torch.tensor(class_weight).to(device)
            self.base_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=class_weight)
        elif self.loss_name == 'SoftmaxCrossEntropyLoss':
            if ignored_index is None:
                ignored_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
            self.base_loss_fn = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignored_index, reduction='none')
        elif self.loss_name == 'FocalLoss':
            self.base_loss_fn = FocalLoss(gamma=kwargs['gamma'], alpha=kwargs['alpha'])
        elif self.loss_name == 'DiceLoss':
            self.base_loss_fn = DiceLoss(class_weight=class_weight)
        elif self.loss_name == 'ExpLoss':
            self.base_loss_fn = ExpLoss(class_weight=class_weight, **kwargs)
        elif self.loss_name == 'MSELoss':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        elif self.loss_name == 'SmoothL1Loss':
            self.base_loss_fn = nn.SmoothL1Loss(reduction='none')
        elif self.loss_name == 'L1Loss':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        else:
            self.logger.critical(f"Unsupported loss function: '{self.loss_name}'. Supported losses: {SUPPORTED_LOSSES}")
        self.base_loss_fn.to(device)

    def __call__(self, output_logits, target, weight=None):
        if weight is None and self.weight_type in ['pyramid', 'sample']:
            self.logger.critical('weight_type %s needs weight as input')
            exit(1)
        if self.loss_name == 'ExpLoss':
            loss = self.base_loss_fn(output_logits, target, weight=weight)
        else:
            loss = self.base_loss_fn(output_logits, target)
            if weight is not None:
                weight = weight.expand_as(loss)
                loss *= weight
        if self.reduction is None:
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            self.logger.critical('Unrecognized reduction method: % s' % self.reduction)
            exit(1)


def get_pyramid_weights(label_map,
                        pyramid_size_threshold,
                        pyramid_weight_interval,
                        pyramid_ignored_index,
                        pyramid_ignored_index_weight,
                        **kwargs):
    # Now only support 2 classes
    label_map = label_map.type(torch.float32)
    pyramid_weights = []
    for label_map_sample in label_map:
        # no pyramid weights
        if torch.sum(label_map_sample) > pyramid_size_threshold:
            weight = torch.where(label_map_sample == pyramid_ignored_index,
                                 torch.ones_like(label_map_sample) * pyramid_ignored_index_weight,
                                 torch.ones_like(label_map_sample) * (sum(pyramid_weight_interval) / 2))
        else:
            last_map = label_map_sample.unsqueeze(0)
            summary_map = label_map_sample.unsqueeze(0)
            while torch.sum(last_map) > pyramid_size_threshold / 20:
                last_map = 1 - torch.nn.MaxPool3d(3, 1, padding=1)(1 - last_map)
                summary_map += last_map
            summary_map = torch.squeeze(summary_map, 0)
            top = torch.max(summary_map)
            interval = pyramid_weight_interval[1] - pyramid_weight_interval[0]
            weight = torch.where(label_map_sample > 0,
                                 interval * summary_map / top + pyramid_weight_interval[0],
                                 torch.ones_like(label_map_sample))
        pyramid_weights.append(weight)
    pyramid_weights = torch.stack(pyramid_weights)
    return pyramid_weights


def get_loss_fns(config, device, logger: logging.Logger):
    """
    Returns the loss functions based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    if config.get('train') is None or config['train'].get('losses') is None:
        logger.critical('Could not find loss method in configuration file')
    loss_configs = config['train']['losses']

    loss_fns = []
    for loss_conf in loss_configs:
        loss_fns.append(
            LossWrapper(num_classes=config['model']['num_classes'], logger=logger, device=device, **loss_conf))
    loss_fns = nn.ModuleList(loss_fns)
    return loss_fns
