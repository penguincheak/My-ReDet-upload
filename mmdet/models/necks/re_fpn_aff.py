import e2cnn.nn as enn
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from e2cnn import gspaces
from mmcv.cnn import constant_init, kaiming_init, xavier_init

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


def conv1x1(inplanes, out_planes, stride=1, padding=0):
    """1x1 convolution"""
    in_type = FIELD_TYPE['regular'](gspace, inplanes)

    out_type = FIELD_TYPE['regular'](gspace, out_planes)
    return enn.R2Conv(in_type, out_type, 1,
                      padding=padding,
                      stride=stride,
                      bias=False,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r,
                      initialize=False)


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


def ennAdaptiveAvgPool(inplanes, outputsize):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseAdaptiveAvgPool(in_type, outputsize)


def ennAdaptiveMaxPool(inplanes, outputsize):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseAdaptiveMaxPool(in_type, outputsize)


# ReChannelAtt used normal conv
class ReChannel_normal_conv_v2(nn.Module):
    def __init__(self, inchannels, r=2):
        super(ReChannel_normal_conv_v2, self).__init__()
        self.inchannels = inchannels
        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.kernel_num = inchannels // Orientation
        self.conv1 = nn.Conv2d(self.kernel_num, self.kernel_num // r, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.kernel_num // r, self.kernel_num, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, c, _, _ = x.size()

        y1 = self.__avg_pool(x)
        featatt1 = []
        for i in range(Orientation):
            feat = y1[:, i::Orientation, :, :]
            feat = self.conv2(self.relu(self.conv1(feat)))
            if i == 0:
                featatt1 = feat
            else:
                featatt1 = torch.cat((featatt1, feat), 1)

        featatt2 = []
        y2 = self.__max_pool(x)
        for i in range(Orientation):
            feat = y2[:, i:: Orientation, :, :]
            feat = self.conv2(self.relu(self.conv1(feat)))
            if i == 0:
                featatt2 = feat
            else:
                featatt2 = torch.cat((featatt2, feat), 1)

        att = []
        for i in range(self.kernel_num):
            feat1 = featatt1[:, i::self.kernel_num, :, :]
            feat2 = featatt2[:, i::self.kernel_num, :, :]
            feat = feat1 + feat2
            if i == 0:
                att = feat
            else:
                att = torch.cat((att, feat), 1)

        return self.sigmoid(att)


'''
===================AFF=======================
'''
class DAF(nn.Module):
    '''
    直接相加 DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=32, r=2):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        # xo = x * wei2 + residual * (1 - wei2)
        # return xo
        return wei2



class ReiAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=256, r=2):
        super(ReiAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        self.local_att_bn1 = ennBatchNorm(inter_channels)
        self.local_att_relu = ennReLU(inter_channels, inplace=True)
        self.local_att_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        self.local_att_bn2 = ennBatchNorm(channels)

        # # 全局注意力
        # self.global_pool = ennAdaptiveAvgPool(channels, 1)
        # self.global_att_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        # self.global_att_bn1 = ennBatchNorm(inter_channels)
        # self.global_att_relu = ennReLU(inter_channels, inplace=True)
        # self.global_att_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        # self.global_att_bn2 = ennBatchNorm(channels)

        # 第二次本地注意力
        self.local_att2_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        self.local_att2_bn1 = ennBatchNorm(inter_channels)
        self.local_att2_relu = ennReLU(inter_channels, inplace=True)
        self.local_att2_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        self.local_att2_bn2 = ennBatchNorm(channels)

        # # 第二次全局注意力
        # self.global_pool2 = ennAdaptiveAvgPool(channels, 1)
        # self.global_att2_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        # self.global_att2_bn1 = ennBatchNorm(inter_channels)
        # self.global_att2_relu = ennReLU(inter_channels, inplace=True)
        # self.global_att2_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        # self.global_att2_bn2 = ennBatchNorm(channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att_bn2(self.local_att_conv2(self.local_att_relu(
            self.local_att_bn1(self.local_att_conv1(xa))
        )))
        # xg = self.global_att_bn2(self.global_att_conv2(self.global_att_relu(
        #     self.global_att_bn1(self.global_att_conv1(self.global_pool(xa)))
        # )))
        # xlg = xl + xg
        xlg = xl
        wei = self.sigmoid(xlg.tensor)
        xa.tensor = x.tensor * wei + residual.tensor * (1 - wei)

        xl2 = self.local_att2_bn2(self.local_att2_conv2(self.local_att2_relu(
            self.local_att2_bn1(self.local_att2_conv1(xa))
        )))
        # xg2 = self.global_att2_bn2(self.global_att2_conv2(self.global_att2_relu(
        #     self.global_att2_bn1(self.global_att2_conv1(self.global_pool2(xa)))
        # )))
        # xlg2 = xl2 + xg2
        xlg2 = xl2
        wei2 = self.sigmoid(xlg2.tensor)
        # xo = x * wei2 + residual * (1 - wei2)
        # return xo
        return wei2

    def evaluate_output_shape(self, input_shape):
        return input_shape


class ReiAFF_single(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=256, r=2):
        super(ReiAFF_single, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        self.local_att_bn1 = ennBatchNorm(inter_channels)
        self.local_att_relu = ennReLU(inter_channels, inplace=True)
        self.local_att_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        self.local_att_bn2 = ennBatchNorm(channels)

        # 全局注意力
        self.global_pool = ennAdaptiveAvgPool(channels, 1)
        self.global_att_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        self.global_att_bn1 = ennBatchNorm(inter_channels)
        self.global_att_relu = ennReLU(inter_channels, inplace=True)
        self.global_att_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        self.global_att_bn2 = ennBatchNorm(channels)

        # 第二次本地注意力
        self.local_att2_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        self.local_att2_bn1 = ennBatchNorm(inter_channels)
        self.local_att2_relu = ennReLU(inter_channels, inplace=True)
        self.local_att2_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        self.local_att2_bn2 = ennBatchNorm(channels)

        # 第二次全局注意力
        self.global_pool2 = ennAdaptiveAvgPool(channels, 1)
        self.global_att2_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        self.global_att2_bn1 = ennBatchNorm(inter_channels)
        self.global_att2_relu = ennReLU(inter_channels, inplace=True)
        self.global_att2_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        self.global_att2_bn2 = ennBatchNorm(channels)

        # 第三次本地注意力
        self.local_att3_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        self.local_att3_bn1 = ennBatchNorm(inter_channels)
        self.local_att3_relu = ennReLU(inter_channels, inplace=True)
        self.local_att3_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        self.local_att3_bn2 = ennBatchNorm(channels)

        # 第三次全局注意力
        self.global_pool3 = ennAdaptiveAvgPool(channels, 1)
        self.global_att3_conv1 = convnxn(channels, inter_channels, kernel_size=1, stride=1)
        self.global_att3_bn1 = ennBatchNorm(inter_channels)
        self.global_att3_relu = ennReLU(inter_channels, inplace=True)
        self.global_att3_conv2 = convnxn(inter_channels, channels, kernel_size=1, stride=1)
        self.global_att3_bn2 = ennBatchNorm(channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att_bn2(self.local_att_conv2(self.local_att_relu(
            self.local_att_bn1(self.local_att_conv1(xa))
        )))
        xg = self.global_att_bn2(self.global_att_conv2(self.global_att_relu(
            self.global_att_bn1(self.global_att_conv1(self.global_pool(xa)))
        )))
        xlg = xl + xg
        wei = self.sigmoid(xlg.tensor)
        xa.tensor = x.tensor * wei + residual.tensor * (1 - wei)

        xl2 = self.local_att2_bn2(self.local_att2_conv2(self.local_att2_relu(
            self.local_att2_bn1(self.local_att2_conv1(xa))
        )))
        xg2 = self.global_att2_bn2(self.global_att2_conv2(self.global_att2_relu(
            self.global_att2_bn1(self.global_att2_conv1(self.global_pool2(xa)))
        )))
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2.tensor)

        xl3 = self.local_att3_bn2(self.local_att3_conv2(self.local_att3_relu(
            self.local_att3_bn1(self.local_att3_conv1(xa))
        )))
        xg3 = self.global_att3_bn2(self.global_att3_conv2(self.global_att3_relu(
            self.global_att3_bn1(self.global_att3_conv1(self.global_pool3(xa)))
        )))
        xlg3 = xl3 + xg3
        wei3 = self.sigmoid(xlg3.tensor)
        # xo = x * wei2 + residual * (1 - wei2)
        # return xo
        return wei3

    def evaluate_output_shape(self, input_shape):
        return input_shape


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=32, r=2):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

'''
===================AFF=======================
'''


# 通道注意力 CBAM 等变卷积实现
class ChannelAttention(enn.EquivariantModule):

    def __init__(self, channel, r=2):
        super(ChannelAttention, self).__init__()

        self.__avg_pool = ennAdaptiveAvgPool(channel, 1)
        self.__max_pool = ennAdaptiveMaxPool(channel, 1)

        self.conv1 = conv1x1(channel, channel // r)
        self.relu = ennReLU(channel // r)
        self.conv2 = conv1x1(channel // r, channel)
        self.__sigmoid = nn.Sigmoid()


    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.conv2(self.relu(self.conv1(y1)))

        y2 = self.__max_pool(x)
        y2 = self.conv2(self.relu(self.conv1(y2)))

        y = self.__sigmoid(y1.tensor + y2.tensor)
        return y

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
class ReFPN_AFF(nn.Module):

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
        super(ReFPN_AFF, self).__init__()
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

        self.channel_attentions = nn.ModuleList()
        # self.channel_attentions2 = nn.ModuleList()
        # self.channel_attention = ChannelAttention(out_channels)

        self.lateral_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.iAFF_list = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            iaff = ReiAFF(int(out_channels))
            self.iAFF_list.append(iaff)

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

            # channel_att = ReChannel_normal_conv_v2(out_channels)
            # channel_att2 = ChannelAttention(out_channels)

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            # self.channel_attentions.append(channel_att)
            # self.channel_attentions2.append(channel_att2)

            self.lateral_convs.append(l_conv)
            self.up_samples.append(up_sample)
            self.fpn_convs.append(fpn_conv)

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
        # 用1*1卷积把通道变为fpn_out
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # for i in range(len(laterals)):
        #     att = self.channel_attentions1[i](laterals[i])
        #     laterals[i].tensor = laterals[i].tensor * att

        # build top-down path
        # 特征融合 用一个特征图
        used_backbone_levels = len(laterals)

        # att = self.channel_attention(laterals[used_backbone_levels - 1])
        # laterals[used_backbone_levels - 1].tensor = laterals[used_backbone_levels - 1].tensor * att

        for i in range(used_backbone_levels - 1, 0, -1):
            # laterals[i - 1] += F.interpolate(
            #     laterals[i], scale_factor=2, mode='nearest')
            # att = self.channel_attentions[i](laterals[i].tensor)
            # laterals[i].tensor = laterals[i].tensor * att

            x = laterals[i - 1]
            y = self.up_samples[i](laterals[i])
            wei = self.iAFF_list[i](x, y)

            laterals[i - 1].tensor = laterals[i - 1].tensor * wei + y.tensor * (1 - wei)

        # for i in range(len(laterals)):
        #     att = self.channel_attentions2[i](laterals[i])
        #     laterals[i].tensor = laterals[i].tensor * att


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

        # 8个方向，8个特征图

        return tuple(outs)
