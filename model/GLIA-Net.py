import torch

from model.global_positioning import RoiPoolingGlobalPositioning, GlobalPositioningAdaptor, GlobalPositioningLoss
from model.layers import get_norm_layer


class GLIANet(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 with_global=False, global_type='attention_map', global_out_channels=2,
                 f_maps=64, norm_type='bn', **kwargs):
        super(GLIANet, self).__init__()
        if isinstance(f_maps, int):
            # use 4 levels(3 down_sample layers) in the encoder path
            f_maps = [f_maps * 2 ** k for k in range(4)]
        assert len(f_maps) > 1
        self.with_global = with_global

        self.encoder_blocks = torch.nn.ModuleList()
        self.encoder_blocks.append(EncodeBlock(in_channels, f_maps[0], down_sample=False, norm_type=norm_type))
        for i in range(1, len(f_maps)):
            self.encoder_blocks.append(EncodeBlock(f_maps[i - 1], f_maps[i], norm_type=norm_type))
        if with_global:
            global_localizer_channels = f_maps[0] * 2
            if global_type == 'attention_map':
                self.global_localizer = RoiPoolingGlobalPositioning(
                    in_channels, global_localizer_channels, EncodeBlock, global_pooling=False, norm_type=norm_type)
            else:
                raise AttributeError('Unrecognized global type %s' % global_type)
            self.localizer_adaptor = torch.nn.ModuleList(
                [GlobalPositioningAdaptor(global_localizer_channels, f_maps[i]) for i in range(len(f_maps) - 1)])
            self.localizer_loss = GlobalPositioningLoss(
                global_localizer_channels, global_out_channels, unit_shape=False, norm_type=norm_type)
        self.decoder_blocks = torch.nn.ModuleList()
        for i in range(len(f_maps) - 1, 0, -1):
            self.decoder_blocks.append(DecodeBlock(f_maps[i], f_maps[i - 1], f_maps[i - 1], norm_type=norm_type))

        self.output_conv = torch.nn.Conv3d(f_maps[0], out_channels, 3, padding=1)

    def forward(self, inputs):
        if self.with_global:
            local_inputs, global_inputs, patch_location_bbox = inputs
            global_localizer_feature = self.global_localizer(global_inputs, patch_location_bbox)
            global_localizer_logits = self.localizer_loss(global_localizer_feature)
            global_localizer_logits = torch.squeeze(torch.squeeze(torch.squeeze(global_localizer_logits, -1), -1), -1)
        else:
            local_inputs = inputs
        net = local_inputs
        skip_connections = []
        for i in range(len(self.encoder_blocks)):
            net = self.encoder_blocks[i](net)
            skip_connections.append(net)
        skip_connections.pop()
        if self.with_global:
            for i in range(len(skip_connections)):
                adaptive_shape = skip_connections[i].shape[2:]
                skip_connections[i] = \
                    self.localizer_adaptor[i](global_localizer_feature, adaptive_shape) * skip_connections[i]
        for i in range(len(self.decoder_blocks)):
            net = self.decoder_blocks[i](net, skip_connections.pop())
        net = self.output_conv(net)
        if self.with_global:
            return net, global_localizer_logits
        else:
            return net


class EncodeBlock(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks=1, down_sample=True, norm_type='bn'):
        super(EncodeBlock, self).__init__()
        if down_sample:
            self.add_module('max_pool', torch.nn.MaxPool3d(2))
        for i in range(num_blocks):
            self.add_module('residual_block%d' % (i + 1),
                            ResidualBlock(out_channels if i else in_channels, out_channels, norm_type))


class DecodeBlock(torch.nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=1, up_sample=True, norm_type='bn'):
        super(DecodeBlock, self).__init__()
        self.up_sample = up_sample
        if up_sample:
            self.de_conv = torch.nn.ConvTranspose3d(
                in_channels, in_channels, 3, stride=2, padding=1, output_padding=1, bias=False)
            self.bn = get_norm_layer(norm_type, in_channels + skip_channels)
            self.lrelu = torch.nn.LeakyReLU(inplace=True)
        self.residual_blocks = torch.nn.Sequential()
        self.residual_blocks.add_module('residual_block1',
                                        ResidualBlock(in_channels + skip_channels, out_channels, norm_type))
        for i in range(1, num_blocks):
            self.residual_blocks.add_module('residual_block%d' % (i + 1),
                                            ResidualBlock(out_channels, out_channels, norm_type))

    def forward(self, inputs, skip_connection):
        if self.up_sample:
            net = self.de_conv(inputs)
        else:
            net = inputs
        if skip_connection is not None:
            net = torch.cat([net, skip_connection], 1)
        net = self.lrelu(self.bn(net))
        net = self.residual_blocks(net)
        return net


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='bn'):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels // 2, 1, padding=0, bias=False)
        self.bn1 = get_norm_layer(norm_type, out_channels // 2)
        self.lrelu1 = torch.nn.LeakyReLU(inplace=True)
        self.conv2 = torch.nn.Conv3d(out_channels // 2, out_channels // 2, 3, padding=1, bias=False)
        self.bn2 = get_norm_layer(norm_type, out_channels // 2)
        self.lrelu2 = torch.nn.LeakyReLU(inplace=True)
        self.conv3 = torch.nn.Conv3d(out_channels // 2, out_channels, 1, padding=0, bias=False)
        self.bn3 = get_norm_layer(norm_type, out_channels)
        self.lrelu3 = torch.nn.LeakyReLU(inplace=True)
        if in_channels != out_channels:
            self.need_res_conv = True
            self.res_conv = torch.nn.Conv3d(in_channels, out_channels, 1, padding=0, bias=False)
        else:
            self.need_res_conv = False

    def forward(self, inputs):
        if self.need_res_conv:
            residual = self.res_conv(inputs)
        else:
            residual = inputs
        net = self.lrelu1(self.bn1(self.conv1(inputs)))
        net = self.lrelu2(self.bn2(self.conv2(net)))
        net = self.conv3(net)
        net = self.lrelu3(self.bn3(net + residual))
        return net

