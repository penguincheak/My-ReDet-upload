# model settings
model = dict(
    type='SSReDet_2',
    pretrained='work_dirs/ReResNet_pretrain/re_resnet50_c8_batch256-25b16846.pth',
    backbone=dict(
        type='ReResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='ReSSP_Re',
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
work_dir = 'work_dirs/ReDet_re50_ressp_3x_dior'
load_from = None
resume_from = None
workflow = [('train', 1)]

# ReDet 12epoch
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

# ssp_re2 min 9 max 12
# AP50: 71.68
# 'airplane': 0.7104967248937426, 'airport': 0.8038998294726049,
# 'baseballfield': 0.7196688690507584, 'basketballcourt': 0.8120551096203019,
# 'bridge': 0.51943432549341, 'chimney': 0.8071921101557618,
# 'dam': 0.6676339428010358, 'Expressway-Service-area': 0.7588615539450011,
# 'Expressway-toll-station': 0.7830618425372143, 'harbor': 0.5833785135768735,
# 'golffield': 0.7773732535719606, 'groundtrackfield': 0.8152993697425,
# 'overpass': 0.5906420584820065, 'ship': 0.8069187318422814,
# 'stadium': 0.7235122929183267, 'storagetank': 0.6991446994306576,
# 'tenniscourt': 0.8136414850479086, 'trainstation': 0.6508335097873393,
# 'vehicle': 0.4839105507068588, 'windmill': 0.8094650305547373

# ssp without re min 9 max 12
# AP50: 64.45
# 'airplane': 0.6704586079749901, 'airport': 0.7790266185471537,
# 'baseballfield': 0.7025961107259459, 'basketballcourt': 0.7965137847128198,
# 'bridge': 0.38383287294051194, 'chimney': 0.7222216956452506,
# 'dam': 0.5931232274384486, 'Expressway-Service-area': 0.5559790011591038,
# 'Expressway-toll-station': 0.5337795613690687, 'harbor': 0.5555060062224266,
# 'golffield': 0.7092282372665217, 'groundtrackfield': 0.7654865327871622,
# 'overpass': 0.5166724636150418, 'ship': 0.7929920624990676,
# 'stadium': 0.6740951157465636, 'storagetank': 0.6136809921015317,
# 'tenniscourt': 0.8034413863797643, 'trainstation': 0.59906716480508,
# 'vehicle': 0.42041430364452437, 'windmill': 0.7011204618649701

# ssp_re2 without resem min 9 max 12
# AP50: 71.37
# 'airplane': 0.7125325121200448, 'airport': 0.7971578282673101,
# 'baseballfield': 0.7200325655118999, 'basketballcourt': 0.812834518014619,
# 'bridge': 0.5313113357594267, 'chimney': 0.8058969279133492, 'dam': 0.665125580466877,
# 'Expressway-Service-area': 0.7749189748247495,
# 'Expressway-toll-station': 0.7716792539964604, 'harbor': 0.5836056074197222,
# 'golffield': 0.7676110817713335, 'groundtrackfield': 0.8323963343914658,
# 'overpass': 0.5947906665666232, 'ship': 0.807802288661389,
# 'stadium': 0.7175406593036302, 'storagetank': 0.6250907592998654,
# 'tenniscourt': 0.812812953254989, 'trainstation': 0.6432144499891338,
# 'vehicle': 0.4873543405647893, 'windmill': 0.8111748586772857

# ssp_re2 without ressm min 9 max 12
# AP50: 71.17
# 'airplane': 0.6292182371164116, 'airport': 0.8004106423232584,
# 'baseballfield': 0.7202470328720223, 'basketballcourt': 0.8138677057544438,
# 'bridge': 0.5260831598317036, 'chimney': 0.8115530022759053, 'dam': 0.6953665851396772,
# 'Expressway-Service-area': 0.7704313427280285,
# 'Expressway-toll-station': 0.7723499750745161, 'harbor': 0.5867126520472175,
# 'golffield': 0.7768125395657335, 'groundtrackfield': 0.8209111101584062,
# 'overpass': 0.5904127197515481, 'ship': 0.8077367030957264,
# 'stadium': 0.7206399162049685, 'storagetank': 0.624375439494719,
# 'tenniscourt': 0.8132330426666905, 'trainstation': 0.652611106361288,
# 'vehicle': 0.48857150029951907, 'windmill': 0.8120059690344498

# ori
# 'airplane': 0.6317045843887453, 'airport': 0.8002458316872858, 'baseballfield': 0.7840165727323941, 'basketballcourt': 0.8135195463781925, 'bridge': 0.5184648372097338, 'chimney': 0.8131365897711914, 'dam': 0.6915580238264528, 'Expressway-Service-area': 0.7848346081570399, 'Expressway-toll-station': 0.7659723597721493, 'harbor': 0.5819192835231635, 'golffield': 0.7862035874094089, 'groundtrackfield': 0.8295736862969052, 'overpass': 0.5898425175809012, 'ship': 0.8083026796859083, 'stadium': 0.7000223372192274, 'storagetank': 0.6269212060128367, 'tenniscourt': 0.813077826414868, 'trainstation': 0.6610010896812475, 'vehicle': 0.48776166433540796, 'windmill': 0.8106670074105335}
# AP50: 71.49


# ReAFF
# 'airplane': 0.7202136177865048, 'airport': 0.7958646065924493, 'baseballfield': 0.7746162192624668, 'basketballcourt': 0.81231775622801, 'bridge': 0.5165325121367382, 'chimney': 0.806934224951704, 'dam': 0.6762054860818981, 'Expressway-Service-area': 0.7584504617789025, 'Expressway-toll-station': 0.7527531393832101, 'harbor': 0.5764150970119041, 'golffield': 0.7770130371429547, 'groundtrackfield': 0.8060389095738016, 'overpass': 0.5936266780047333, 'ship': 0.8080610643844661, 'stadium': 0.6760509840945444, 'storagetank': 0.70092951173755, 'tenniscourt': 0.8125112922266547, 'trainstation': 0.6361255879133016, 'vehicle': 0.4927946048308006, 'windmill': 0.8100810634552686}
# AP50: 71.52

# ReAFF + ori
# 'airplane': 0.7197767392694533, 'airport': 0.8010899576279746, 'baseballfield': 0.7858281897210392, 'basketballcourt': 0.8124260461606867, 'bridge': 0.5282337161942432, 'chimney': 0.8148788719914049, 'dam': 0.6933891692196679, 'Expressway-Service-area': 0.7893047264702561, 'Expressway-toll-station': 0.776253409012341, 'harbor': 0.6098713957202959, 'golffield': 0.7834911905798798, 'groundtrackfield': 0.8240608129447131, 'overpass': 0.6108843273445713, 'ship': 0.8093717473364304, 'stadium': 0.6908256645024863, 'storagetank': 0.7084302084470097, 'tenniscourt': 0.8129377761255838, 'trainstation': 0.6524487917534579, 'vehicle': 0.49497627851042164, 'windmill': 0.8107693918203457}
# AP50: 72.65

# ReSSP_re2
# 'airplane': 0.7134312843532599, 'airport': 0.7951365905969884, 'baseballfield': 0.7184694227332825, 'basketballcourt': 0.8144153353399997, 'bridge': 0.53057477017814, 'chimney': 0.8076946141028329, 'dam': 0.7144119056711805, 'Expressway-Service-area': 0.7665706690874692, 'Expressway-toll-station': 0.7689237432982843, 'harbor': 0.5800156586228331, 'golffield': 0.764477361790093, 'groundtrackfield': 0.8211973236516769, 'overpass': 0.5917058983141612, 'ship': 0.8081431698475919, 'stadium': 0.6871182939658877, 'storagetank': 0.7003189551911388, 'tenniscourt': 0.8130268054590365, 'trainstation': 0.648530798165918, 'vehicle': 0.4874497370645034, 'windmill': 0.8113216242428888}
# AP50: 71.71

# ReSSP_re2 + ReAFFPN
# 'airplane': 0.7204468862554394, 'airport': 0.7971282062912391, 'baseballfield': 0.7815356122098368, 'basketballcourt': 0.8135512686420332, 'bridge': 0.5289354342599486, 'chimney': 0.8101252138063595, 'dam': 0.7143821141498962, 'Expressway-Service-area': 0.7734820643156033, 'Expressway-toll-station': 0.7804516652791249, 'harbor': 0.6040231129002196, 'golffield': 0.778998139089451, 'groundtrackfield': 0.8184588660141483, 'overpass': 0.6156286603314999, 'ship': 0.8089451079167604, 'stadium': 0.6878240857685334, 'storagetank': 0.708824754827923, 'tenniscourt': 0.8127240928021489, 'trainstation': 0.6472903768142109, 'vehicle': 0.4963431513932299, 'windmill': 0.8118210628138418}
# AP50: 72.55
