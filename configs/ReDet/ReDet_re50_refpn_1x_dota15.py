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
        type='SharedFCBBoxHeadRbbox_OAtt',
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
        type='SharedFCBBoxHeadRbbox_OAtt',
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
data_root = '/home/penguin/Experiments/ReDet/data/dota1_5-split-1024/'
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
work_dir = 'work_dirs/ReDet_re50_refpn_1x_dota15'
load_from = None
resume_from = None
workflow = [('train', 1)]

# OBB
# map: 0.6685876697941695
# classaps:  [79.20329861 82.81152908 51.91645747 71.40502646 52.38051243 75.72773116
#  80.92049489 90.83033489 75.8138614  68.6393801  49.28574084 72.02986006
#  73.36013455 70.55389787 63.33062659 11.53138528]

# HBB
# map: 0.6765991141966352
# classaps:  [79.51441367 82.63204656 53.80948876 69.82043539 52.75941583 75.6409354
#  87.81692604 90.83033489 75.8138614  68.77949859 49.10840864 71.6478249
#  75.56893056 75.17287145 58.28604779 15.35714286]

# SPAtt-enn on ResNet only
# mAP: 0.6665228098070406
# ap of each class: plane:0.7823634533676103, baseball-diamond:0.8181739690044355,
# bridge:0.5313116500960511, ground-track-field:0.6542826085498773,
# small-vehicle:0.5234704521025183, large-vehicle:0.7535303407888947, ship:0.8627589187915015,
# tennis-court:0.9082976529033104, basketball-court:0.7954969926963339,
# storage-tank:0.6811771086000501, soccer-ball-field:0.5059919726851866, roundabout:0.7258239889193961,
# harbor:0.7380118244369003, swimming-pool:0.6455167466007603,
# helicopter:0.6082871474996955, container-crane:0.12987012987012986

# ReFPN + ReCBAM on fpn OBB
# mAP: 0.6695528552639038
# ap of each class: plane:0.7911824088158215, baseball-diamond:0.8348277752904405,
# bridge:0.5414686716691625, ground-track-field:0.6522842229327402,
# small-vehicle:0.5246401175361903, large-vehicle:0.7587433431997259,
# ship:0.8729240362347218, tennis-court:0.9071799878801823,
# basketball-court:0.7963360704276792, storage-tank:0.6705498105459183,
# soccer-ball-field:0.5758910837086323, roundabout:0.7338454117031048,
# harbor:0.7326149697971076, swimming-pool:0.6439197274628271,
# helicopter:0.6082562288363884, container-crane:0.06818181818181818

# ReFPN + ReCBAM on fpn HBB
# mAP: 0.6797397787385514
# ap of each class: plane:0.7937631437845352, baseball-diamond:0.8331916437509539,
# bridge:0.5541436629545409, ground-track-field:0.6494761694023724,
# small-vehicle:0.527792633737356, large-vehicle:0.7568211213370065,
# ship:0.8806719512670068, tennis-court:0.9071799878801823,
# basketball-court:0.7912337788492387, storage-tank:0.6726883861281606,
# soccer-ball-field:0.5840988864444492, roundabout:0.7304019520458366,
# harbor:0.7558954516472268, swimming-pool:0.7367917174956368,
# helicopter:0.5735867995385996, container-crane:0.12809917355371903

# ReResNet + Att OBB
# mAP: 0.6666479216948102
# ap of each class: plane:0.7774550637540493, baseball-diamond:0.8102357674333308,
# bridge:0.529111258664581, ground-track-field:0.6844886785937613,
# small-vehicle:0.5183693146863612, large-vehicle:0.745131900726395,
# ship:0.8670756137643336, tennis-court:0.9084070230997394,
# basketball-court:0.8074308189895316, storage-tank:0.6802721755239166,
# soccer-ball-field:0.47225117986531107, roundabout:0.7260194175035254,
# harbor:0.7257531889441319, swimming-pool:0.7035894948013471,
# helicopter:0.619866759857556, container-crane:0.09090909090909091

# ReResNet + Att HBB
# mAP: 0.6712984705464705
# ap of each class: plane:0.7827731341709996, baseball-diamond:0.8058229363247386,
# bridge:0.5445318967409427, ground-track-field:0.6827203851675065,
# small-vehicle:0.5224954754834663, large-vehicle:0.7428368205415586,
# ship:0.8790070171131106, tennis-court:0.9084070230997394,
# basketball-court:0.8015008580547085, storage-tank:0.6826170620121412,
# soccer-ball-field:0.47225117986531107, roundabout:0.715049921891214,
# harbor:0.7593889103310335, swimming-pool:0.7494276324206194,
# helicopter:0.6010361846173455, container-crane:0.09090909090909091

# ReFPN + ReCBAM 包括最后一层
# mAP: 0.6608988235265415
# ap of each class: plane:0.7888635586699924, baseball-diamond:0.8133089595647967,
# bridge:0.5159470769393179, ground-track-field:0.6703316480644711,
# small-vehicle:0.5224965475629575, large-vehicle:0.7900108501103529,
# ship:0.8736397253069235, tennis-court:0.9076345273719852,
# basketball-court:0.7690183294271702, storage-tank:0.6677623242609748,
# soccer-ball-field:0.5044800041354774, roundabout:0.7343222909401239,
# harbor:0.746825702341371, swimming-pool:0.6505433025732436,
# helicopter:0.6044543144134928, container-crane:0.014742014742014743

# OAtt on RPN HBB
# mAP: 0.6782905559794026
# ap of each class: plane:0.7920631158823221, baseball-diamond:0.8147290380409479,
# bridge:0.5347511513527394, ground-track-field:0.637144577960345,
# small-vehicle:0.5267436296604546, large-vehicle:0.7553623487352767,
# ship:0.8805764479155116, tennis-court:0.9077114320045233,
# basketball-court:0.8220047182715684, storage-tank:0.6734867569708773,
# soccer-ball-field:0.5298518025884174, roundabout:0.7270190901995157,
# harbor:0.7599669205307894, swimming-pool:0.7471852411191144,
# helicopter:0.5843138005201087, container-crane:0.15973882391792837

# ReCHAtt on FPN All layers
# mAP: 0.6752113184073041
# ap of each class: plane:0.7857096921235383, baseball-diamond:0.8218246203823745,
# bridge:0.5253017799601267, ground-track-field:0.692943217160769,
# small-vehicle:0.5227034208620673, large-vehicle:0.7556937547746405,
# ship:0.8702352699812678, tennis-court:0.9086904526488304,
# basketball-court:0.8145912657867117, storage-tank:0.6838440774276526,
# soccer-ball-field:0.5582244361489446, roundabout:0.7271835240518906,
# harbor:0.7329200378516895, swimming-pool:0.6542815402848512,
# helicopter:0.6249501832467044, container-crane:0.12428382182480543

# ReSPAtt on FPN All layers
# mAP: 0.6618905996828195
# ap of each class: plane:0.7877021340471373, baseball-diamond:0.8257967923098647,
# bridge:0.5344354107793898, ground-track-field:0.7002149508712505,
# small-vehicle:0.5234243614905205, large-vehicle:0.7555348334964999,
# ship:0.809527801828673, tennis-court:0.9080304029141772,
# basketball-court:0.7369304721399742, storage-tank:0.6851210395001386,
# soccer-ball-field:0.5120947124710125, roundabout:0.7280434776523382,
# harbor:0.7416427746318536, swimming-pool:0.6607557857803397,
# helicopter:0.6474799852295314, container-crane:0.033514659782408265

# ReFPN + IoUSmoothL1Loss
# mAP: 0.6710100571267757
# ap of each class: plane:0.785196183970729, baseball-diamond:0.8194783353043946,
# bridge:0.5202484175015225, ground-track-field:0.6945927514857555,
# small-vehicle:0.5229087003342366, large-vehicle:0.7522580359487671,
# ship:0.8711002597805535, tennis-court:0.908696224181029,
# basketball-court:0.7977659213109779, storage-tank:0.6842860729134318,
# soccer-ball-field:0.535901884008874, roundabout:0.7318663622286458,
# harbor:0.7356334638923242, swimming-pool:0.6497153625183101,
# helicopter:0.6356038477397701, container-crane:0.09090909090909091

# ReCHAtt normal conv v2 not share sigmoid
# mAP: 0.6650905368350486
# ap of each class: plane:0.7854287070992308, baseball-diamond:0.8189440923274075,
# bridge:0.5283085959035165, ground-track-field:0.6937719848285004,
# small-vehicle:0.5201686999181722, large-vehicle:0.7572770961579118,
# ship:0.8091378699714229, tennis-court:0.9063806529881404,
# basketball-court:0.789983852710785, storage-tank:0.6882955491005469,
# soccer-ball-field:0.5163857739530098, roundabout:0.7193602684059589,
# harbor:0.6810690021122846, swimming-pool:0.6955731581519237,
# helicopter:0.6315609141904636, container-crane:0.09980237154150198

# ReChatt not share sigmoid
# mAP: 0.6658666333580632
# ap of each class: plane:0.7814885546687634, baseball-diamond:0.7952170485840523,
# bridge:0.5332537237618324, ground-track-field:0.6950814660726735,
# small-vehicle:0.5210589381586865, large-vehicle:0.7573195858725177,
# ship:0.8090609379049838, tennis-court:0.9063049770132398,
# basketball-court:0.7637119626922801, storage-tank:0.6760896546754334,
# soccer-ball-field:0.5290956688596834, roundabout:0.7178484612729855,
# harbor:0.7412238835316906, swimming-pool:0.6579917434079251,
# helicopter:0.6254995929696797, container-crane:0.14361993428258488








