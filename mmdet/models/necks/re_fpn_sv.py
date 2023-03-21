import e2cnn.nn as enn
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from e2cnn import gspaces
from mmcv.cnn import constant_init, kaiming_init, xavier_init
from ..losses.attention_loss import AttentionLoss

from ..registry import NECKS

# Set default Orientation=8, .i.e, the group C8
# One can change it by passing the env Orientation=xx
Orientation = 8
# keep similar computation or similar params
# One can change it by passing the env fixparams=True
fixparams = False
if 'Orientation' in os.environ:
    Orientation = int(os.environ['Orientation'])
if 'fixparams' in os.environ:
    fixparams = True

gspace = gspaces.Rot2dOnR2(N=Orientation)


def regular_feature_type(gspace: gspaces.GSpace, planes: int):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()
    if fixparams:
        planes *= math.sqrt(N)
    planes = planes / N
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int):
    """ build a trivial feature map with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())

    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}


def convnxn(inplanes, outplanes, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    out_type = FIELD_TYPE['regular'](gspace, outplanes)
    return enn.R2Conv(in_type, out_type, kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias,
                      dilation=dilation,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r, )


def ennReLU(inplanes, inplace=True):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.ReLU(in_type, inplace=inplace)


def ennInterpolate(inplanes, scale_factor, mode='nearest', align_corners=False):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.R2Upsampling(in_type, scale_factor, mode=mode, align_corners=align_corners)


def ennMaxPool(inplanes, kernel_size, stride=1, padding=0):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseMaxPool(in_type, kernel_size=kernel_size, stride=stride, padding=padding)


def build_conv_layer(cfg, *args, **kwargs):
    layer = convnxn(*args, **kwargs)
    return layer


def build_norm_layer(cfg, num_features, postfix=''):
    in_type = FIELD_TYPE['regular'](gspace, num_features)
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)


def ennBatchNorm(depth):
    in_type = FIELD_TYPE['regular'](gspace, depth)
    return enn.InnerBatchNorm(in_type)


def ennAdaptiveAvgPool(inplanes, out):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseAdaptiveAvgPool(in_type, output_size=out)


class LevelAttentionModel(nn.Module):

    def __init__(self, num_features_in, feature_size=256, stride=1):
        super(LevelAttentionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1, stride=stride)
        self.act1 = nn.ReLU(feature_size)

        self.conv5 = nn.Conv2d(feature_size, 1, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv5(out)
        out_attention = F.sigmoid(out)

        return out_attention


class ReASPP(enn.EquivariantModule):
    def __init__(self, in_channels = 1024, depth = 256):
        super(ReASPP, self).__init__()
        self.depth = depth

        self.conv_1x1_1 = convnxn(in_channels, depth, kernel_size=1)
        self.bn_conv_1x1_1 = ennBatchNorm(depth)
        self.relu_1x1_1 = ennReLU(depth)

        self.conv_3x3_1 = convnxn(in_channels, depth,kernel_size=3, stride=1, padding=6, dilation=12)
        self.bn_conv_3x3_1 = ennBatchNorm(depth)
        self.relu_3x3_1 = ennReLU(depth)

        self.conv_3x3_2 = convnxn(in_channels, depth, kernel_size=3, stride=1, padding=12, dilation=24)
        self.bn_conv_3x3_2 = ennBatchNorm(depth)
        self.relu_3x3_2 = ennReLU(depth)

        self.conv_3x3_3 = convnxn(in_channels, depth, kernel_size=3, stride=1, padding=18, dilation=36)
        self.bn_conv_3x3_3 = ennBatchNorm(depth)
        self.relu_3x3_3 = ennReLU(depth)

        self.avg_pool = ennAdaptiveAvgPool(in_channels, 1)

        self.conv_1x1_2 = convnxn(in_channels, depth, kernel_size=1)
        self.bn_conv_1x1_2 = ennBatchNorm(depth)
        self.relu_1x1_2 = ennReLU(depth)

        self.conv_1x1_3 = convnxn(5 * depth, depth, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = ennBatchNorm(depth)
        self.relu_1x1_3 = ennReLU(depth)

        # 用有监督的注意力机制计算heatmaps
        # generate heatmaps Ak
        self.att_h2 = LevelAttentionModel(num_features_in=depth, stride=1)

        self.att_h3 = LevelAttentionModel(num_features_in=depth, stride=2)

        self.att_h4 = LevelAttentionModel(num_features_in=depth, stride=4)

        self.att_h5 = LevelAttentionModel(num_features_in=depth, stride=8)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 4*256, h/4, w/4))

        feature_map_h = feature_map.size()[2] # (== h/4)
        feature_map_w = feature_map.size()[3] # (== w/4)

        out_1x1 = self.relu_1x1_1(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))).tensor # (shape: (batch_size, 256, h/4, w/4))
        out_3x3_1 = self.relu_3x3_1(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))).tensor # (shape: (batch_size, 256, h/4, w/4))
        out_3x3_2 = self.relu_3x3_2(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))).tensor # (shape: (batch_size, 256, h/4, w/4))
        out_3x3_3 = self.relu_3x3_3(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))).tensor # (shape: (batch_size, 256, h/4, w/4))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = self.relu_1x1_2(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img.tensor, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/4, w/4))

        out = torch.cat((out_1x1, out_3x3_1), 1)
        out = torch.cat((out, out_3x3_2), 1)
        out = torch.cat((out, out_3x3_3), 1)
        out = torch.cat((out, out_img), 1)

        # out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/4, w/4))
        out = enn.GeometricTensor(out, FIELD_TYPE['regular'](gspace, self.depth * 5))
        out = self.relu_1x1_3(self.bn_conv_1x1_3(self.conv_1x1_3(out))).tensor # (shape: (batch_size, 256, h/4, w/4))

        heatmap2 = self.att_h2(out)
        heatmap3 = self.att_h3(out)
        heatmap4 = self.att_h4(out)
        heatmap5 = self.att_h5(out)

        return [heatmap2, heatmap3, heatmap4, heatmap5]

    def evaluate_output_shape(self, input_shape):
        return input_shape

class ConvModule(enn.EquivariantModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.in_type = enn.FieldType(gspace, [gspace.regular_repr] * in_channels)
        self.out_type = enn.FieldType(gspace, [gspace.regular_repr] * out_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = padding
        self.groups = groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            if conv_cfg != None and conv_cfg['type'] == 'ORConv':
                norm_channels = int(norm_channels * 8)
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = ennReLU(out_channels, inplace=self.inplace)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        # kaiming_init(self.conv, nonlinearity=nonlinearity)
        # if self.with_norm:
        #     constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x

    def evaluate_output_shape(self, input_shape):
        return input_shape


@NECKS.register_module
class ReFPN_SV(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(ReFPN_SV, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)

            up_sample = ennInterpolate(out_channels, 2)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.up_samples.append(up_sample)
            self.fpn_convs.append(fpn_conv)

        # generate ASPP features
        self.aspp = ReASPP(out_channels * (self.backbone_end_level - self.start_level), out_channels)

        # calculate attention loss
        self.attentionloss = AttentionLoss()

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.max_pools = nn.ModuleList()
        self.relus = nn.ModuleList()

        used_backbone_levels = len(self.lateral_convs)
        if self.num_outs > used_backbone_levels:
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    self.max_pools.append(ennMaxPool(out_channels, 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                for i in range(used_backbone_levels + 1, self.num_outs):
                    self.relus.append(ennReLU(out_channels))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # 到这里通道数都是256
        # upsample to the bottom features
        used_backbone_levels = len(laterals)
        att_heatmaps = []
        for i in range(used_backbone_levels - 1, -1, -1):
            if i == used_backbone_levels - 1:
                att_heatmaps = F.interpolate(laterals[i].tensor,
                                            size=(laterals[0].tensor.shape[2:]), mode='nearest')
            else:
                att_heatmaps = torch.cat((att_heatmaps, F.interpolate(laterals[i].tensor,
                                            size=(laterals[0].tensor.shape[2:]), mode='nearest')), 1)

        # concatenate laterals
        # att_heatmaps = torch.cat([i for i in att_heatmaps], 1)

        att_heatmaps = enn.GeometricTensor(att_heatmaps, FIELD_TYPE['regular'](gspace, self.out_channels * used_backbone_levels))

        # generate ASPP features
        att_heatmaps = self.aspp(att_heatmaps)

        # generate SEM outputs
        for i in range(0, used_backbone_levels):
            attF = att_heatmaps[i].expand_as(laterals[i].tensor)
            attF = attF * laterals[i].tensor
            laterals[i].tensor = attF + laterals[i].tensor

        # generate SSM outputs
        # build top-down path
        for i in range(used_backbone_levels - 1, 0, -1):
            attF = F.interpolate(att_heatmaps[i], [att_heatmaps[i - 1].size(2),
                        att_heatmaps[i - 1].size(3)], mode='nearest')
            attF = attF * att_heatmaps[i - 1]
            # laterals[i - 1] += F.interpolate(
            #     laterals[i], scale_factor=2, mode='nearest')
            lateral = self.up_samples[i](laterals[i])
            lateral.tensor *= attF.expand_as(laterals[i - 1].tensor)
            laterals[i - 1] += lateral

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(self.max_pools[i](outs[-1]))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](self.relus[i](outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # convert to tensor
        outs = [out.tensor for out in outs]

        return tuple(outs), att_heatmaps
