# model settings
model = dict(
    type='ReDet',
    pretrained='work_dirs/ReResNet_pretrain/re_resnet50_c8_batch256-25b16846.pth',
    backbone=dict(
        type='ReResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='ReFPN_AFF',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
        reg_class_agnostic=True,
        with_module=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        # loss_bbox=dict(type='IoUSmoothL1Loss', beta=1.0, loss_weight=1.0)),
    rbbox_roi_extractor=dict(
        type='RboxSingleRoIExtractor',
        roi_layer=dict(type='RiRoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    rbbox_head = dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.05, 0.05, 0.1, 0.1, 0.05],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    )
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssignerCy',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssignerRbbox',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomRbboxSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ])
test_cfg = dict(
    rpn=dict(
        # TODO: test nms 2000
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr = 0.05, nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img = 2000)
)
# dataset settings
dataset_type = 'HRSCL1Dataset'
data_root = 'data/HRSC2016/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Train/HRSC_L1_train.json',
        img_prefix=data_root + 'Train/images/',
        img_scale=(800, 512), #[(600, 384), (800, 512), (1000, 640)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Test/HRSC_L1_test.json',
        img_prefix=data_root + 'Test/images/',
        img_scale=(800, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Test/HRSC_L1_test.json',
        img_prefix=data_root + 'Test/images/',
        # img_scale=[(800, 512), (800, 600)],
        img_scale=(800, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        # rotate_test_aug=True,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 33])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 36
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/ReDet_re50_refpn_3x_hrsc2016'
load_from = None
resume_from = None
workflow = [('train', 1)]

# VOC2007 metrics
# AP50: 90.46	AP75: 89.46	 mAP: 70.41
# AP50: 90.42	AP75: 89.36	 mAP: 71.14
# AP50: 90.45	AP75: 89.29	 mAP: 71.41

# VOC2007 metrics RePANet
# AP50: 90.55	AP75: 89.80	 mAP: 71.77

# repanet + rechatt before fpn + respatt after td + iou smooth l1 loss
# AP50: 90.55	AP75: 89.53	 mAP: 71.87

# repanet + rechatt before fpn
# AP50: 90.53	AP75: 89.43	 mAP: 71.26

# ReSEP
# AP50: 90.42	AP75: 89.67	 mAP: 71.35

# VOC2007 metrics RePANet + SPAtt-enn
# AP50: 90.54	AP75: 89.03	 mAP: 71.46

# VOC2007 metrics SPAtt-normal
# AP50: 90.35	AP75: 88.76	 mAP: 70.66

# VOC2007 metrics SPAtt-enn
# AP50: 90.50	AP75: 89.15	 mAP: 70.89

# VOC2007 metrics ecaLayer
# AP50: 90.45	AP75: 89.26	 mAP: 71.02

# LightECALayer
# AP50: 90.39	AP75: 89.42	 mAP: 71.23

# SElayer-normal
# AP50: 90.50	AP75: 89.13	 mAP: 71.26

# ReSELayer
# AP50: 90.55	AP75: 89.23	 mAP: 71.11

# ReChannelAtt
# AP50: 90.49	AP75: 89.48	 mAP: 71.38

# SElayer-normal + resep (共享参数)
# AP50: 90.34	AP75: 89.20	 mAP: 71.39

# SElayer-normal + resep (每层单独设置卷积)
# AP50: 90.48	AP75: 89.25	 mAP: 70.66

# ReSELayer + SPAtt-enn
# AP50: 90.48	AP75: 89.31	 mAP: 71.39

# ReSPAtt + ReCHAtt
# AP50: 90.55	AP75: 89.58	 mAP: 71.73

# ReSPAtt + ReCHAtt + RePANet
# AP50: 90.35	AP75: 89.13	 mAP: 70.82
# AP50: 90.39	AP75: 89.22	 mAP: 70.61

# ReFPN-Attention only on fuse path
# AP50: 90.46	AP75: 89.22	 mAP: 70.84
# ReFPN-Attention on FPN
# AP50: 90.58	AP75: 89.44	 mAP: 71.68

# RePANet-Att only on FPN 注意力加在FPN上，只多一条bottom-up
# AP50: 90.49	AP75: 89.55	 mAP: 71.65
# AP50: 90.52	AP75: 89.66	 mAP: 71.63

# RePANet-Att 在FPN上和自底向上前都用ch-enn sp-enn
# AP50: 90.57	AP75: 88.99	 mAP: 70.17
# AP50: 90.56	AP75: 89.33	 mAP: 70.22

# RePANet-Att 在自底向上前用ch-enn sp-enn
# AP50: 90.57	AP75: 89.56	 mAP: 71.00

# RePANet only on FPN 注意力加在FPN上，只多一条bottom-up + ResNet-Att
# AP50: 90.47	AP75: 89.44	 mAP: 70.68

# Orientation Att on ReResNet
# AP50: 90.48	AP75: 89.28	 mAP: 71.04
# 只用一个注意力卷积的实现
# AP50: 90.50	AP75: 89.64	 mAP: 71.64

# CBAM -> OAtt
# AP50: 90.40	AP75: 89.18	 mAP: 70.68

# OAtt -> CBAM
# AP50: 90.47	AP75: 89.22	 mAP: 71.47

# ReDCM and ReBCD
# AP50: 90.63	AP75: 89.65	 mAP: 72.06
# AP50: 90.64	AP75: 89.56	 mAP: 72.05

# OAtt on RPN
# AP50: 90.54	AP75: 89.21	 mAP: 71.91

# ReFPN + ReAtt + RPN_OAtt
# AP50: 90.53	AP75: 89.22	 mAP: 70.91

# iou_smooth_l1 loss on stage 1
# AP50: 90.53	AP75: 89.62	 mAP: 71.85
# on stage 1 and 2
# AP50: 90.41	AP75: 89.73	 mAP: 72.44

# iou_smooth_l1 loss + RPN_OAtt
# AP50: 90.52	AP75: 89.84	 mAP: 71.38

# RechannelAtt share sigmoid
# AP50: 90.51	AP75: 89.58	 mAP: 72.20
# AP50: 90.62	AP75: 89.69	 mAP: 71.63

# RechannelAtt not share sigmoid
# AP50: 90.46	AP75: 89.48	 mAP: 71.09

# RechannelAtt normal conv share sigmoid
# AP50: 90.51	AP75: 89.44	 mAP: 71.74

# RechannelAtt normal conv_v2 share sigmoid
# AP50: 90.63	AP75: 89.68	 mAP: 72.61
# AP50: 90.63	AP75: 89.67	 mAP: 71.98
# on all layers
# AP50: 90.53	AP75: 89.46	 mAP: 72.16

# RechannelAtt normal conv_v2 share sigmoid on RePANet_Top-Down
# AP50: 90.54	AP75: 89.54	 mAP: 71.67
# on RePANet_Bottom-Up
# AP50: 90.55	AP75: 89.60	 mAP: 71.82

# ReChAtt_shared_sigmoid on FPN + IoUSmoothl1
# AP50: 90.62	AP75: 89.68	 mAP: 72.59

# ReChAtt_normal conv_v2 not share sigmoid + OAtt on RPN
# AP50: 90.62	AP75: 89.34	 mAP: 72.03

# iou smooth l1 loss (right) on stage 1
# AP50: 90.61	AP75: 89.86	 mAP: 71.59

# iou smooth l1 loss 复现完全 on stage1
# AP50: 90.49	AP75: 89.56	 mAP: 71.11 错了

# AP50: 90.44	AP75: 89.85	 mAP: 71.60 对的
# AP50: 90.47	AP75: 89.71	 mAP: 72.49


# rechatt normal conv v2 sigmoid both
# AP50: 90.56	AP75: 89.57	 mAP: 72.76
# rechatt normal conv v2 sigmoid before
# AP50: 90.60	AP75: 89.64	 mAP: 71.49
# plus IoU Smooth L1
# AP50: 90.52	AP75: 89.79	 mAP: 71.73
# plus IoU Smooth L1 * IoU + (1 - IoU) * smoothl1
# AP50: 90.52	AP75: 89.60	 mAP: 72.24
# plus IoU Smooth L1 * IoU + (1 - IoU) * smoothl1 and rechatt not on lastlayer
# AP50: 90.56	AP75: 89.92	 mAP: 71.94


# refpn + aff
# AP50: 90.56	AP75: 89.70	 mAP: 72.09

# refpn + aff + rechatt before fpn
# AP50: 90.58	AP75: 89.67	 mAP: 71.37

# refpn + aff + rechatt after fpn
# AP50: 90.57	AP75: 89.68	 mAP: 71.40

# refpn + aff + rechatt before and after fpn
# AP50: 90.60	AP75: 89.65	 mAP: 71.45

# redet + normal chatt
# AP50: 90.31	AP75: 80.84	 mAP: 68.93

# redet + normal chatt r=1
# AP50: 90.42 AP75: 87.96 mAP: 70.74

# redet + normal chatt without relu
# AP50: 90.21	AP75: 88.21	 mAP: 69.92

# ReAFF one iter
# AP50: 90.48	AP75: 89.55	 mAP: 71.88

# ReAFF without local att
# AP50: 90.53	AP75: 89.61	 mAP: 71.72

# ReAFF without global att
# AP50: 90.47	AP75: 89.51	 mAP: 71.20

