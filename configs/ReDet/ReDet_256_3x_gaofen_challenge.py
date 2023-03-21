# model settings
model = dict(
    type='ReDet',
    pretrained='work_dirs/ReResNet_pretrain/re_resnet50_c8_batch256-25b16846.pth',
    backbone=dict(
        type='ReResNet',
        depth=50,
        stem_channels=64,
        base_channels=64,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='ReFPN',
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
        num_classes=11,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
        reg_class_agnostic=True,
        with_module=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    rbbox_roi_extractor=dict(
        type='RboxSingleRoIExtractor',
        roi_layer=dict(type='RiRoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    rbbox_head=dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=11,
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
dataset_type = 'GaoFen_challengeDataset'
data_root = 'data/gaofen-challenge/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/GaoFen_train.json',
        img_prefix=data_root + 'train/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train/GaoFen_train.json',
        img_prefix=data_root + 'train/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/GaoFen_test.json',
        img_prefix=data_root + 'test/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
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
checkpoint_config = dict(interval=12)
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
work_dir = 'work_dirs/ReDet_256_3x_gaofen_challenge'
load_from = None
resume_from = None
workflow = [('train', 1)]

# 预训练权重 FPN out = 256
#  {'other': 0.676395714668169, 'A220': 0.2556367295322729, 'Boeing737': 0.40932010523714196, 'A321': 0.5452044507532721, 'Boeing787': 0.4779288678514212, 'Boeing747': 0.7335959451347263, 'A330': 0.6759484845173404, 'Boeing777': 0.06806712881792415, 'A350': 0.7600070309367115, 'ARJ21': 0.0}
# AP50: 46.02

# 预训练权重 PANet out = 256
# 'other': 0.7460404278205534, 'A220': 0.2925036050981209, 'Boeing737': 0.2982500822981098, 'A321': 0.6829387869449504, 'Boeing787': 0.5373134771630563, 'Boeing747': 0.8440585723994424, 'A330': 0.7882647574771617, 'Boeing777': 0.05350193665983139, 'A350': 0.8359151507300077, 'ARJ21': 0.0
# ap50: 50.02

# 用预训练权重，FPN out=512
# AP50: 52.05
# 'other': 0.7322893084640549, 'A220': 0.29639868577850514, 'Boeing737': 0.305777727248052, 'A321': 0.6778709496380613, 'Boeing787': 0.6391775479782871, 'Boeing747': 0.8875544180652848, 'A330': 0.738230857538756, 'Boeing777': 0.10245310245310246, 'A350': 0.8250093061966609, 'ARJ21': 0.0

# 用预训练权重，FPN out=1024
# AP50: 53.85
# 'other': 0.7337432699578629, 'A220': 0.3340486201730221, 'Boeing737': 0.33801247918524013, 'A321': 0.7127217630132076, 'Boeing787': 0.6472965688573163, 'Boeing747': 0.8869760414583974, 'A330': 0.7237021438226303, 'Boeing777': 0.11248477788955363, 'A350': 0.8958515873980133, 'ARJ21': 0.0}

# AP50: 51.79 FPN out=512 ECA
# 'other': 0.7032006636123165, 'A220': 0.36498281307947805, 'Boeing737': 0.30808478343362067, 'A321': 0.6985321732098881, 'Boeing787': 0.5826237824534506, 'Boeing747': 0.8523060794123145, 'A330': 0.7432237451433722, 'Boeing777': 0.11268398943434028, 'A350': 0.8134258291942352, 'ARJ21': 0.0

# AP50: 49.49 FPN out=512 BCAM 通道注意力
# 'other': 0.6515762580539611, 'A220': 0.2781235808150979, 'Boeing737': 0.2917135961383749, 'A321': 0.6497930765900032, 'Boeing787': 0.5656436974536792, 'Boeing747': 0.8563932921620644, 'A330': 0.7122998529282837, 'Boeing777': 0.1117907614427336, 'A350': 0.8316030570390754, 'ARJ21': 0.0

# repanet out=512
#'other': 0.6889060652322025, 'A220': 0.3156284579513821, 'Boeing737': 0.2930997999711298, 'A321': 0.6570991109274358, 'Boeing787': 0.6142417099323189, 'Boeing747': 0.8442112408218062, 'A330': 0.7360011315579171, 'Boeing777': 0.10943865094062724, 'A350': 0.8440529058750005, 'ARJ21': 0.0


# 重新跑了一下256
# AP50: 50.92
# 'other': 0.7042149425751433, 'A220': 0.3043789825286894, 'Boeing737': 0.26257622514113965, 'A321': 0.6665845088994105, 'Boeing787': 0.591062579277757, 'Boeing747': 0.8491874451171957, 'A330': 0.7848521289695132, 'Boeing777': 0.11654456327985739, 'A350': 0.8128427818889633, 'ARJ21': 0.0
# AP50: 53.07
# {'other': 0.6992078518356615, 'A220': 0.36155991097511103, 'Boeing737': 0.35035723974369104, 'A321': 0.7074541228489254, 'Boeing787': 0.6075138769149298, 'Boeing747': 0.8828224423172688, 'A330': 0.747886340544884, 'Boeing777': 0.1129032258064516, 'A350': 0.8368545832396952, 'ARJ21': 0.0

# ECA 256
# AP50: 48.30
# 'other': 0.692442602254934, 'A220': 0.3017078896349102, 'Boeing737': 0.2235326360613702, 'A321': 0.7265793813423896, 'Boeing787': 0.6433059306111844, 'Boeing747': 0.7580649645053668, 'A330': 0.6522510773999848, 'Boeing777': 0.05650718994143951, 'A350': 0.7757366128369294, 'ARJ21': 0.0
