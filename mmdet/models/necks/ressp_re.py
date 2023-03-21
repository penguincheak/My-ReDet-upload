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


class ReASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)
    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 5, 1)):
        super(ReASPP, self).__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = convnxn(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = ennAdaptiveAvgPool(inplanes=in_channels, out=1)
        self.relu = ennReLU(out_channels)

    def forward(self, x):
        # x 是GTensor
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(self.relu(self.aspp[aspp_idx](inp)).tensor)
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        out = enn.GeometricTensor(out, regular_feature_type(gspace, out.shape[1]))
        return out


class ReCAM(nn.Module):
    def __init__(self, inplanes, reduction_ratio=1, fpn_lvl=4):
        super(ReCAM, self).__init__()
        self.fpn_lvl = fpn_lvl
        self.inplanes = inplanes

        self.dila_conv1 = convnxn(inplanes * fpn_lvl // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=3, stride=1, padding=1)
        self.dila_conv2 = ReASPP(inplanes // reduction_ratio, inplanes // (4 * reduction_ratio))
        self.dila_conv3 = convnxn(inplanes // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=3, stride=1, padding=1)
        self.dila_conv4 = ennBatchNorm(inplanes // reduction_ratio)
        self.dila_conv5 = ennReLU(inplanes // reduction_ratio, inplace=False)

        self.sigmoid = nn.Sigmoid()
        self.upsample_cfg = dict(mode='nearest')
        self.down_conv = nn.ModuleList()
        self.att_conv = nn.ModuleList()
        self.Interpolate = nn.ModuleList()
        for i in range(self.fpn_lvl):
            self.att_conv.append(convnxn(inplanes // reduction_ratio,
                                           Orientation,
                                           kernel_size=3,
                                           stride=1,  # 2 ** i
                                           padding=1))
            if i == 0:
                down_stride = 1
            else:
                down_stride = 2
            self.down_conv.append(
                convnxn(inplanes // reduction_ratio, inplanes // reduction_ratio, kernel_size=3, stride=down_stride,
                          padding=1))
            if i > 0:
                self.Interpolate.append(ennInterpolate(inplanes // reduction_ratio,
                                                    2 ** i))


    def forward(self, x):

        multi_feats = [x[0].tensor]

        for i in range(1, len(x)):
            pyr_feats_2x = self.Interpolate[i - 1](x[i])
            multi_feats.append(pyr_feats_2x.tensor)

        multi_feats = torch.cat(multi_feats, 1)
        multi_feats = enn.GeometricTensor(multi_feats, regular_feature_type(gspace, x[0].tensor.shape[1] * self.fpn_lvl))

        lvl_fea = self.dila_conv1(multi_feats)
        lvl_fea = self.dila_conv2(lvl_fea)
        lvl_fea = self.dila_conv3(lvl_fea)
        lvl_fea = self.dila_conv4(lvl_fea)
        lvl_fea = self.dila_conv5(lvl_fea)

        multi_atts = []

        for i in range(self.fpn_lvl):
            lvl_fea = self.down_conv[i](lvl_fea)
            lvl_att = self.att_conv[i](lvl_fea)
            multi_atts.append(lvl_att.tensor)

        # visualization

        # for i in range(self.fpn_lvl):  # self.fpn_lvl
        #     att = (multi_atts[i].detach().cpu().numpy()[0])
        #     # att /= np.max(att)
        #     #att = np.power(att, 0.8)
        #     att = att * 255
        #     att = att.astype(np.uint8).transpose(1, 2, 0)
        #    # att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        #     mmcv.imshow(att)
        #     cv2.waitKey(0)

        return multi_atts



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
class ReSSP_Re(nn.Module):

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
        super(ReSSP_Re, self).__init__()
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
        self.CAM = ReCAM(out_channels)

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


        att_heatmaps = self.CAM(laterals)
        # att_heatmaps是没有sigmoid的，结果相加之后再sigmoid

        # resem
        for i in range(self.start_level, self.backbone_end_level):
            # att = []
            # for j in range(self.out_channels // Orientation):
            #     if j == 0:
            #         att = att_heatmaps[i]
            #     else:
            #         att = torch.cat((att, att_heatmaps[i]), 1)
            att = []
            for k in range(Orientation):
                for j in range(self.out_channels // Orientation):
                    if j == 0 and k == 0:
                        att = att_heatmaps[i][:, 0, :, :].reshape(att_heatmaps[i].shape[0], 1, att_heatmaps[i].shape[2], att_heatmaps[i].shape[3])
                    else:
                        att = torch.cat((att, att_heatmaps[i][:, k, :, :].reshape(att_heatmaps[i].shape[0], 1, att_heatmaps[i].shape[2], att_heatmaps[i].shape[3])), 1)

            laterals[i].tensor = (1 + torch.sigmoid(att)) * laterals[i].tensor

        # generate SSM outputs
        # build top-down path
        # 特征融合
        # ressm
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] += F.interpolate(
            #     laterals[i], scale_factor=2, mode='nearest')
            att_2x = F.interpolate(torch.sigmoid(att_heatmaps[i]), size=prev_shape)
            att_2x = torch.sigmoid(att_heatmaps[i - 1]) * att_2x
            # att_insec = []
            # for j in range(self.out_channels // Orientation):
            #     if j == 0:
            #         att_insec = att_2x
            #     else:
            #         att_insec = torch.cat((att_insec, att_2x), 1)
            att_insec = []
            for k in range(Orientation):
                for j in range(self.out_channels // Orientation):
                    if j == 0 and k == 0:
                        att_insec = att_2x[:, 0, :, :].reshape(att_2x.shape[0], 1, att_2x.shape[2], att_2x.shape[3])
                    else:
                        att_insec = torch.cat((att_insec, att_2x[:, k, :, :].reshape(att_2x.shape[0], 1, att_2x.shape[2], att_2x.shape[3])), 1)

            laterals[i - 1].tensor += att_insec * self.up_samples[i](laterals[i]).tensor

        # # build top-down path
        # # 特征融合
        # used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     # laterals[i - 1] += F.interpolate(
        #     #     laterals[i], scale_factor=2, mode='nearest')
        #     laterals[i - 1] += self.up_samples[i](laterals[i])

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

        # 计算8个方向att_heatmaps的和
        att_sum = []
        for i in range(len(att_heatmaps)):
            for j in range(att_heatmaps[i].shape[1]):
                if(j == 0):
                    att = att_heatmaps[i][:, j, :, :]
                else:
                    att = att + att_heatmaps[i][:, j, :, :]
            att_sum.append(torch.sigmoid(att).reshape(att.shape[0], 1, att.shape[1], att.shape[2]))

        return tuple(outs), att_sum
