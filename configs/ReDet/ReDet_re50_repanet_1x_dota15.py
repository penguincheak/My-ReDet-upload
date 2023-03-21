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
        type='RePANet',
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
        num_classes=17,
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
        num_classes=17,
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
        score_thr=0.05, nms=dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img=2000)
)
# dataset settings
dataset_type = 'DOTA1_5Dataset_v2'
data_root = 'data/dota1_5-split-1024/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA1_5_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA1_5_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test1024/DOTA1_5_test1024.json',
        img_prefix=data_root + 'test1024/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
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
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/ReDet_re50_repanet_1x_dota15'
load_from = None
resume_from = None
workflow = [('train', 1)]

# OBB
# map: 0.6685876697941695
# classaps:  [79.20329861 82.81152908 51.91645747 71.40502646 52.38051243 75.73
#  80.92049489 90.83033489 75.8138614  68.6393801  49.28574084 72.02986006
#  73.36013455 70.55389787 63.33062659 11.53138528]

# HBB
# map: 0.6765991141966352
# classaps:  [79.51441367 82.63204656 53.80948876 69.82043539 52.75941583 75.6409354
#  87.81692604 90.83033489 75.8138614  68.77949859 49.10840864 71.6478249
#  75.56893056 75.17287145 58.28604779 15.35714286]

# ReFPN + AFF
# mAP: 0.6249018616487368
# classaps: plane:0.7697427155704373, baseball-diamond:0.7554967724587462,
# bridge:0.5049463243577065, ground-track-field:0.6338227207086703, small-vehicle:0.5202101199621048,
# large-vehicle:0.7415510417101229, ship:0.8074485348470888, tennis-court:0.9088274044795785,
# basketball-court:0.7675844893727338, storage-tank:0.6728686245262873,
# soccer-ball-field:0.450491330485437, roundabout:0.6973216880834944, harbor:0.67322055803727,
# swimming-pool:0.6308138470069342, helicopter:0.4640836147731786, container-crane:0.0

# repanet OBB  0.02 lr
# mAP: 0.673482810146529
# classaps: plane:0.7875277297821011, baseball-diamond:0.8428186756875421,
# bridge:0.5208152875507591, ground-track-field:0.6606222916376902, small-vehicle:0.5218664539618689,
# large-vehicle:0.757198335184213, ship:0.8715726798518021, tennis-court:0.9086317722681361,
# basketball-court:0.794039768125189, storage-tank:0.684643173381503,
# soccer-ball-field:0.5341563130485811, roundabout:0.7301231030083737, harbor:0.7386116537847854,
# swimming-pool:0.6648571836762963, helicopter:0.6401397473589986, container-crane:0.11810079403662291

# Repanet 和 Attention
# mAP: 0.6675750539824221
# classaps: plane:0.7202275337033139, baseball-diamond:0.8328851779228581,
# bridge:0.5398851644229201, ground-track-field:0.6330702196713873, small-vehicle:0.5182931806229294,
# large-vehicle:0.7610523962038325, ship:0.8709460309014545, tennis-court:0.908628373576063,
# basketball-court:0.7870205395038941, storage-tank:0.6838494574157795,
# soccer-ball-field:0.5044467198287188, roundabout:0.72712661922601, harbor:0.7435071410246327,
# swimming-pool:0.7216120747947228, helicopter:0.6610826673326674, container-crane:0.06756756756756757


# normal attention use one orientation in fpn used ECA
# 这个是用参数共享的fuse attention
# mAP: 0.66771717318171
# ap of each class: plane:0.788967646
# classaps: plane:0.7889676463797984, baseball-diamond:0.8333763700952954,
# bridge:0.5367156111812045, ground-track-field:0.661540213833714, small-vehicle:0.5232968067552733,
# large-vehicle:0.7606770818352697, ship:0.8690128373960162, tennis-court:0.9017453288864191,
# basketball-court:0.749921374877774, storage-tank:0.6842373205037526,
# soccer-ball-field:0.4796693624916277, roundabout:0.7238270990252997, harbor:0.7501735340130729,
# swimming-pool:0.7042771314506393, helicopter:0.582065760316172, container-crane:0.1339712918660287


# rese in resnet 误
# mAP: 0.6596406008384771
# classaps: plane:0.7846384367040058, baseball-diamond:0.8336545195168586,
# bridge:0.5300390518499388, ground-track-field:0.6539262430799083, small-vehicle:0.5215409753356821,
# large-vehicle:0.7586921723954615, ship:0.8082411207530347, tennis-court:0.907009483653852,
# basketball-court:0.8085279093513396, storage-tank:0.6829842374850634,
# soccer-ball-field:0.4881511830195612, roundabout:0.7084210912918119, harbor:0.736978770209323,
# swimming-pool:0.6509251697851391, helicopter:0.5755035750034614, container-crane:0.10501567398119123
# 加了注意力对小目标识别差

# rese in refpn
# mAP: 0.6561157246326811
# classaps: plane:0.7878682114054566, baseball-diamond:0.7758041284580839, bridge:0.5140498332061665, ground-track-field:0.6402708245790271, small-vehicle:0.522491252506433, large-vehicle:0.7590738753852014, ship:0.8096629412218463, tennis-court:0.9085039780879094, basketball-court:0.7672509228669488, storage-tank:0.6761707839126362, soccer-ball-field:0.4878019983463951, roundabout:0.7319204468552284, harbor:0.7276958109608277, swimming-pool:0.6565644911902958, helicopter:0.6281766405949863, container-crane:0.10454545454545455
# 加在fpn效果差，选择加在resnet中

# RePANet + Att + OBB
# mAP: 0.6676609767874123
# ap of each class: plane:0.7898645709660063, baseball-diamond:0.8407690366609311,
# bridge:0.5360174216537327, ground-track-field:0.648772868848975,
# small-vehicle:0.5239738255618703, large-vehicle:0.7625387752255988,
# ship:0.8711376621674246, tennis-court:0.9083531145948174,
# basketball-court:0.75671417680509, storage-tank:0.6730916505586908,
# soccer-ball-field:0.506704377716478, roundabout:0.7134043144172643,
# harbor:0.7396206518629492, swimming-pool:0.6598751586109706,
# helicopter:0.6441313439125486, container-crane:0.10760667903525047

# RePANet + ReCHAtt on Both TD and BU
# mAP: 0.6711483184103889
# ap of each class: plane:0.7854569728313759, baseball-diamond:0.8212983520798766,
# bridge:0.5263590803559377, ground-track-field:0.6687840547306385,
# small-vehicle:0.5227676444720916, large-vehicle:0.7603721535585511,
# ship:0.8682667153030734, tennis-court:0.9082777111816989,
# basketball-court:0.8115801014902393, storage-tank:0.6811258545223012,
# soccer-ball-field:0.5035403172007689, roundabout:0.7190432258814221,
# harbor:0.7363196042713283, swimming-pool:0.6922643386712521,
# helicopter:0.626856361955062, container-crane:0.10606060606060606

# RePANet + ReCHAtt on TD
# mAP: 0.6695303938177262
# ap of each class: plane:0.7914936547490965, baseball-diamond:0.8100400074898793,
# bridge:0.5237629987298645, ground-track-field:0.6835473161269192,
# small-vehicle:0.5228471972814651, large-vehicle:0.7585676589710714,
# ship:0.8640633031975133, tennis-court:0.9080540806184019,
# basketball-court:0.7952696533865234, storage-tank:0.6846793195006964,
# soccer-ball-field:0.5043703814738288, roundabout:0.7229646518696342,
# harbor:0.7368279015814335, swimming-pool:0.7016979436006031,
# helicopter:0.5955658296546362, container-crane:0.10873440285204991






