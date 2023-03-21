import mmcv
from torch import nn

from .registry import (BACKBONES, NECKS, ROI_EXTRACTORS, SHARED_HEADS, HEADS,
                       LOSSES, DETECTORS)


def _build_module(cfg, registry, default_args):
    # cfg为ConfigDict但继承自dict
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        if obj_type not in registry.module_dict:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
        # 注册model，即把model的名字加入到module_dict中
        obj_type = registry.module_dict[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)

# isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
# type() 不会认为子类是一种父类类型，不考虑继承关系。
# isinstance() 会认为子类是一种父类类型，考虑继承关系。
def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)

# DETECTORS是一个_name为'detector'的Registry的类，用于注册模型(将模型名称添加到名为_model_dict的dict的字典中)
# cfg是model的config
def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
