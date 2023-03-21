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
        type='ReFPN_Attention',
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
        num_classes=21,
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
        num_classes=21,
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
dataset_type = 'DIORDataset'
data_root = 'data/dior/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/DIOR_L1_train.json',
        img_prefix=data_root + 'trainval/images/',
        img_scale=(800, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/DIOR_L1_test.json',
        img_prefix=data_root + 'trainval/images/',
        img_scale=(800, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/DIOR_L1_test.json',
        img_prefix=data_root + 'test/images/',
        img_scale=(800, 800),
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
    step=[8, 11])
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
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/ReDet_re50_refpn_3x_dior'
load_from = None
resume_from = None
workflow = [('train', 1)]

# ReDet 12epoch
# 1
# AP50: 71.49
# 'airplane': 0.6317045843887453, 'airport': 0.8002458316872858,
# 'baseballfield': 0.7840165727323941, 'basketballcourt': 0.8135195463781925,
# 'bridge': 0.5184648372097338, 'chimney': 0.8131365897711914,
# 'dam': 0.6915580238264528, 'Expressway-Service-area': 0.7848346081570399,
# 'Expressway-toll-station': 0.7659723597721493, 'harbor': 0.5819192835231635,
# 'golffield': 0.7862035874094089, 'groundtrackfield': 0.8295736862969052,
# 'overpass': 0.5898425175809012, 'ship': 0.8083026796859083,
# 'stadium': 0.7000223372192274, 'storagetank': 0.6269212060128367,
# 'tenniscourt': 0.813077826414868, 'trainstation': 0.6610010896812475,
# 'vehicle': 0.48776166433540796, 'windmill': 0.8106670074105335
# 2
# {'airplane': 0.6304024623212945, 'airport': 0.7979732254484947,
# 'baseballfield': 0.781336480936979, 'basketballcourt': 0.8140939347485383,
# 'bridge': 0.5155527184906774, 'chimney': 0.8077274133882015,
# 'dam': 0.6601710921421269, 'Expressway-Service-area': 0.7630835166445221,
# 'Expressway-toll-station': 0.777230825637989, 'harbor': 0.5837109883763644,
# 'golffield': 0.7753908544831039, 'groundtrackfield': 0.8263803804784298,
# 'overpass': 0.5955650018476343, 'ship': 0.8078079287326518,
# 'stadium': 0.7057101937049073, 'storagetank': 0.6246783963953664,
# 'tenniscourt': 0.8142969813500377, 'trainstation': 0.6501733384041775,
# 'vehicle': 0.48593166561782253, 'windmill': 0.8080504225123929}
# AP50: 71.13

# iAFF
# AP50: 71.52
#  {'airplane': 0.7202136177865048, 'airport': 0.7958646065924493,
#  'baseballfield': 0.7746162192624668, 'basketballcourt': 0.81231775622801,
#  'bridge': 0.5165325121367382, 'chimney': 0.806934224951704,
#  'dam': 0.6762054860818981, 'Expressway-Service-area': 0.7584504617789025,
#  'Expressway-toll-station': 0.7527531393832101, 'harbor': 0.5764150970119041,
#  'golffield': 0.7770130371429547, 'groundtrackfield': 0.8060389095738016,
#  'overpass': 0.5936266780047333, 'ship': 0.8080610643844661,
#  'stadium': 0.6760509840945444, 'storagetank': 0.70092951173755,
#  'tenniscourt': 0.8125112922266547, 'trainstation': 0.6361255879133016,
#  'vehicle': 0.4927946048308006, 'windmill': 0.8100810634552686}

# re channel attention
# AP50: 71.38
# {'airplane': 0.6334110600328013, 'airport': 0.7999972992871912,
# 'baseballfield': 0.782599293935493, 'basketballcourt': 0.8138757613974441,
# 'bridge': 0.5284059308574834, 'chimney': 0.8127491094993947, 'dam': 0.6855832243601003,
# 'Expressway-Service-area': 0.7578321142468544,
# 'Expressway-toll-station': 0.7750285953433516, 'harbor': 0.5830187404245322,
# 'golffield': 0.7702460042506927, 'groundtrackfield': 0.8326235267970695,
# 'overpass': 0.5926216566562141, 'ship': 0.8073947977517412,
# 'stadium': 0.7025662201633539, 'storagetank': 0.6256763284880836,
# 'tenniscourt': 0.8137758996277917, 'trainstation': 0.6644106769740366,
# 'vehicle': 0.4855061751899855, 'windmill': 0.8087365805694713}