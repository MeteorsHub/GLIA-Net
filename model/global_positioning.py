import logging

import torch

from model.layers import get_norm_layer


class RoiPoolingGlobalPositioning(torch.nn.Module):
    def __init__(self, in_channels, out_channels, encode_block_cls, global_pooling=True, norm_type='bn'):
        super(RoiPoolingGlobalPositioning, self).__init__()
        self.feature_generator = torch.nn.Sequential()
        # stride = 4
        self.feature_generator.add_module('encode_block1', encode_block_cls(
            in_channels, out_channels // 4, down_sample=False, norm_type=norm_type))
        self.feature_generator.add_module('encode_block2', encode_block_cls(
            out_channels // 4, out_channels // 2, down_sample=True, norm_type=norm_type))
        self.feature_generator.add_module('encode_block3', encode_block_cls(
            out_channels // 2, out_channels, down_sample=True, norm_type=norm_type))
        self.feature_generator.add_module('encode_block4', encode_block_cls(
            out_channels, out_channels * 2, down_sample=False, norm_type=norm_type))
        self.feature_generator.add_module('encode_block5', encode_block_cls(
            out_channels * 2, out_channels * 4, down_sample=False, norm_type=norm_type))

        self.roi_adaptive_pooling = torch.nn.AdaptiveMaxPool3d(6)

        self.localizer_generator = torch.nn.Sequential()
        self.localizer_generator.add_module('encode_block1', encode_block_cls(
            out_channels * 4, out_channels * 2, down_sample=False, norm_type=norm_type))
        self.localizer_generator.add_module('encode_block2', encode_block_cls(
            out_channels * 2, out_channels, down_sample=False, norm_type=norm_type))
        if global_pooling:
            self.localizer_generator.add_module('global_avg_pooling', torch.nn.AdaptiveAvgPool3d(1))

    def forward(self, global_inputs, patch_location_bbox):
        net = self.feature_generator(global_inputs)
        # stride 4
        stride_patch_location_bbox = torch.round((patch_location_bbox / 4)).type(torch.int32)
        roi_features = []
        for global_feature, bbox in zip(net, stride_patch_location_bbox):
            # check bbox boundary
            for i in range(3):
                if bbox[i * 2] < 0:
                    bbox[i * 2] = 0
                if bbox[i * 2] >= global_feature.shape[i + 1]:
                    logging.getLogger('ROIPoolingLocalizer').warning('Unexpected bbox %s for feature_shape %s'
                                                                     % (bbox, global_feature.shape))

                    bbox[i * 2] = global_feature.shape[i + 1] - 1
                if bbox[i * 2 + 1] >= global_feature.shape[i + 1]:
                    bbox[i * 2 + 1] = global_feature.shape[i + 1] - 1
                if bbox[i * 2 + 1] < bbox[i * 2]:
                    bbox[i * 2 + 1] = bbox[i * 2]
            roi_feature = global_feature[:, bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1, bbox[4]:bbox[5] + 1]
            roi_features.append(self.roi_adaptive_pooling(roi_feature))
        roi_features = torch.stack(roi_features)
        net = self.localizer_generator(roi_features)
        return net


class GlobalPositioningAdaptor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalPositioningAdaptor, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs, adaptive_shape=None):
        if adaptive_shape is not None:
            net = torch.nn.AdaptiveAvgPool3d(adaptive_shape)(inputs)
        else:
            net = inputs
        net = self.sigmoid(self.conv(net))
        return net


class GlobalPositioningLoss(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, unit_shape=True, norm_type='bn'):
        super(GlobalPositioningLoss, self).__init__()
        if unit_shape:
            self.add_module('conv1', torch.nn.Conv3d(in_channels, in_channels // 2, 1))
            self.add_module('lrelu1', torch.nn.LeakyReLU(inplace=True))
            self.add_module('conv2', torch.nn.Conv3d(in_channels // 2, out_channels, 1))
        else:
            self.add_module('conv1', torch.nn.Conv3d(in_channels, in_channels // 2, 3, padding=1, bias=False))
            self.add_module('bn1', get_norm_layer(norm_type, in_channels // 2))
            self.add_module('lrelu1', torch.nn.LeakyReLU(inplace=True))
            self.add_module('global_max_pool', torch.nn.AdaptiveMaxPool3d(1))
            self.add_module('conv2', torch.nn.Conv3d(in_channels // 2, out_channels, 1))
