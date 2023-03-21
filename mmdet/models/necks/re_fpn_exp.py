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


def conv1x1(inplanes, out_planes, stride=1):
    """1x1 convolution"""
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    out_type = FIELD_TYPE['regular'](gspace, out_planes)
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      bias=False,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r,
                      initialize=False)


def ennReLU(inplanes, inplace=True):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.ReLU(in_type, inplace=inplace)


def ennBatchNorm(depth):
    in_type = FIELD_TYPE['regular'](gspace, depth)
    return enn.InnerBatchNorm(in_type)


def ennInterpolate(inplanes, scale_factor, mode='nearest', align_corners=False):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.R2Upsampling(in_type, scale_factor, mode=mode, align_corners=align_corners)


def ennMaxPool(inplanes, kernel_size, stride=1, padding=0):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseMaxPool(in_type, kernel_size=kernel_size, stride=stride, padding=padding)


def ennAdaptiveAvgPool(inplanes, output_size=1):
    intype = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseAdaptiveAvgPool(intype, output_size)


def ennAdaptiveMaxPool(inplanes, output_size=1):
    intype = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseAdaptiveMaxPool(intype, output_size)


def build_conv_layer(cfg, *args, **kwargs):
    layer = convnxn(*args, **kwargs)
    return layer


def build_norm_layer(cfg, num_features, postfix=''):
    in_type = FIELD_TYPE['regular'](gspace, num_features)
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)


# 空间注意力 CBAM 等变卷积实现
class SpatialAttention(enn.EquivariantModule):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.conv = convnxn(2 * Orientation, Orientation, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_num = x.size(1)
        num_field = int(channel_num / Orientation)
        y = []
        for i in range(0, Orientation):
            feature = x[:, i::8, :, :]
            avgout = torch.mean(feature, dim=1, keepdim=True)
            maxout, _ = torch.max(feature, dim=1, keepdim=True)
            if i != 0:
                y = torch.cat((y, avgout), dim=1)
                y = torch.cat((y, maxout), dim=1)
            else:
                y = torch.cat((avgout, maxout), dim=1)

        y = enn.GeometricTensor(y, enn.FieldType(gspace, 2 * [gspace.regular_repr]))
        y = self.conv(y)
        # sigmoid
        y = self.sigmoid(y.tensor)
        result = []
        for i in range(0, num_field):
            if i == 0:
                result = y
            else:
                result = torch.cat((result, y), dim=1)
        return result

    def evaluate_output_shape(self, input_shape):
        return input_shape

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


class ReDCM_D(nn.Module):
    def __init__(self, inplanes, laterals_len):
        super(ReDCM_D, self).__init__()
        self.inplanes = inplanes
        self.DownSample_list = nn.ModuleList()
        self.laterals_conv_list = nn.ModuleList()
        self.BatchNorm_list = nn.ModuleList()
        self.ReLU_list = nn.ModuleList()
        for i in range(laterals_len):
            Downsample = convnxn(inplanes, inplanes, kernel_size=3, padding=1,
                                 stride=pow(2, laterals_len - i), dilation=pow(2, laterals_len - i - 1))
            lateral_conv = convnxn(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
            BatchNorm = ennBatchNorm(inplanes)
            ReLU = ennReLU(inplanes)
            self.DownSample_list.append(Downsample)
            self.laterals_conv_list.append(lateral_conv)
            self.BatchNorm_list.append(BatchNorm)
            self.ReLU_list.append(ReLU)

        self.conv1 = convnxn(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = ennBatchNorm(inplanes)
        self.relu1 = ennReLU(inplanes)

        self.conv2 = convnxn(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = ennBatchNorm(inplanes)
        self.relu2 = ennReLU(inplanes)

        self.conv3 = convnxn(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = ennBatchNorm(inplanes)
        self.relu3 = ennReLU(inplanes)

        self.conv4 = convnxn(inplanes * (laterals_len + 1), inplanes, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = ennBatchNorm(inplanes)
        self.relu4 = ennReLU(inplanes)

    def forward(self, x, laterals):
        for i in range(len(laterals)):
            laterals[i] = self.ReLU_list[i](self.BatchNorm_list[i](
                self.laterals_conv_list[i](self.DownSample_list[i](laterals[i]))
            ))

        y = self.relu1(self.batchnorm1(self.conv1(x)))
        for i in range(len(laterals)):
            y.tensor = y.tensor * laterals[i].tensor
        y = self.relu2(self.batchnorm2(self.conv2(y)))

        y = y + x
        y = self.relu3(self.batchnorm3(self.conv3(y))).tensor

        for i in range(len(laterals)):
            y = torch.cat((y, laterals[i].tensor), 1)
        y = enn.GeometricTensor(y, FIELD_TYPE['regular'](gspace, self.inplanes * (len(laterals) + 1)))

        y = self.relu4(self.batchnorm4(self.conv4(y)))

        return y
        
        
        
class ReDCM_U(nn.Module):
    def __init__(self, inplanes, laterals_len):
        super(ReDCM_U, self).__init__()
        self.inplanes = inplanes
        self.UpSample_list = nn.ModuleList()
        self.laterals_conv_list = nn.ModuleList()
        self.BatchNorm_list = nn.ModuleList()
        self.ReLU_list = nn.ModuleList()
        for i in range(laterals_len):
            Upsample = ennInterpolate(inplanes, pow(2, i + 1))
            lateral_conv = convnxn(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
            BatchNorm = ennBatchNorm(inplanes)
            ReLU = ennReLU(inplanes)
            self.UpSample_list.append(Upsample)
            self.laterals_conv_list.append(lateral_conv)
            self.BatchNorm_list.append(BatchNorm)
            self.ReLU_list.append(ReLU)

        self.conv1 = convnxn(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = ennBatchNorm(inplanes)
        self.relu1 = ennReLU(inplanes)

        self.conv2 = convnxn(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = ennBatchNorm(inplanes)
        self.relu2 = ennReLU(inplanes)

        self.conv3 = convnxn(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = ennBatchNorm(inplanes)
        self.relu3 = ennReLU(inplanes)

        self.conv4 = convnxn(inplanes * (laterals_len + 1), inplanes, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = ennBatchNorm(inplanes)
        self.relu4 = ennReLU(inplanes)

    def forward(self, x, laterals):
        for i in range(len(laterals)):
            laterals[i] = self.ReLU_list[i](self.BatchNorm_list[i](
                self.laterals_conv_list[i](self.UpSample_list[i](laterals[i]))
            ))

        y = self.relu1(self.batchnorm1(self.conv1(x)))
        for i in range(len(laterals)):
            y.tensor = y.tensor * laterals[i].tensor
        y = self.relu2(self.batchnorm2(self.conv2(y)))

        y = y + x
        y = self.relu3(self.batchnorm3(self.conv3(y))).tensor

        for i in range(len(laterals)):
            y = torch.cat((y, laterals[i].tensor), 1)
        y = enn.GeometricTensor(y, FIELD_TYPE['regular'](gspace, self.inplanes * (len(laterals) + 1)))

        y = self.relu4(self.batchnorm4(self.conv4(y)))

        return y



@NECKS.register_module
class ReFPN_EXP(nn.Module):

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
        super(ReFPN_EXP, self).__init__()
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

        # self.up_samples = nn.ModuleList()
        self.CH_attention = nn.ModuleList()
        self.SP_attention = nn.ModuleList()
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

            ch_att = ChannelAttention(out_channels, 2)
            sp_att = SpatialAttention(3)

            # up_sample = ennInterpolate(out_channels, 2)
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

            # self.up_samples.append(up_sample)
            self.CH_attention.append(ch_att)
            self.SP_attention.append(sp_att)
            self.fpn_convs.append(fpn_conv)

        # BCD1
        self.BCD1_Up_list = nn.ModuleList()
        self.BCD1_Down_list = nn.ModuleList()
        for i in range(3):
            dcm_u = ReDCM_U(out_channels, i + 1)
            self.BCD1_Up_list.append(dcm_u)
            dcm_d = ReDCM_D(out_channels, i + 1)
            self.BCD1_Down_list.append(dcm_d)

        # BCD2
        self.BCD2_Up_list = nn.ModuleList()
        self.BCD2_Down_list = nn.ModuleList()
        for i in range(2):
            dcm_u = ReDCM_U(out_channels, i + 1)
            self.BCD2_Up_list.append(dcm_u)
            dcm_d = ReDCM_D(out_channels, i + 1)
            self.BCD2_Down_list.append(dcm_d)

        # BCD3
        self.BCD3_Up_list = nn.ModuleList()
        self.BCD3_Down_list = nn.ModuleList()
        for i in range(1):
            dcm_u = ReDCM_U(out_channels, i + 1)
            self.BCD3_Up_list.append(dcm_u)
            dcm_d = ReDCM_D(out_channels, i + 1)
            self.BCD3_Down_list.append(dcm_d)

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

        # build top-down path
        # 特征融合
        used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     # laterals[i - 1] += F.interpolate(
        #     #     laterals[i], scale_factor=2, mode='nearest')
        #     laterals[i - 1] += self.up_samples[i](laterals[i])

        for i in range(used_backbone_levels):
            att = self.CH_attention[i](laterals[i])
            laterals[i].tensor = laterals[i].tensor * att
            att = self.SP_attention[i](laterals[i].tensor)
            laterals[i].tensor = laterals[i].tensor * att

        # BCD block
        # BCD1-top-down
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.BCD1_Up_list[used_backbone_levels - 1 - i](
                laterals[i - 1], laterals[i: used_backbone_levels])
        # BCD1-bottom-up
        for i in range(used_backbone_levels - 1):
            laterals[i + 1] = self.BCD1_Down_list[i](laterals[i + 1], laterals[0: i + 1])

        # BCD2-top-down
        for i in range(used_backbone_levels - 2, 0, -1):
            laterals[i - 1] = self.BCD2_Up_list[used_backbone_levels - 2 - i](
                laterals[i - 1], laterals[i: used_backbone_levels - 1])
        # BCD2-bottom-up
        for i in range(used_backbone_levels - 2):
            laterals[i + 1] = self.BCD2_Down_list[i](laterals[i + 1], laterals[0: i + 1])

        # BCD3-top-down
        for i in range(used_backbone_levels - 3, 0, -1):
            laterals[i - 1] = self.BCD3_Up_list[used_backbone_levels - 3 - i](
                laterals[i - 1], laterals[i: used_backbone_levels - 2])
        # BCD3-bottom-up
        for i in range(used_backbone_levels - 3):
            laterals[i + 1] = self.BCD3_Down_list[i](laterals[i + 1], laterals[0: i + 1])

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

        # # 认为设计权重
        # weight = torch.tensor([1., 1., 1., 1., 1., 2., 4., 3.], device=torch.cuda.current_device())
        # weight = weight.view(1, 8, 1, 1)
        # w_sigmoid = nn.Sigmoid()
        # weight = w_sigmoid(weight)
        # weights = []
        # for i in range(int(self.out_channels / Orientation)):
        #     if i == 0:
        #         weights = weight
        #     else:
        #         weights = torch.cat((weights, weight), 1)

        # convert to tensor
        outs = [out.tensor for out in outs]

        # 8个方向，8个特征图

        return tuple(outs)
