default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='CS',
             group='CS',
             name='CS_E20230309_0'
         )
         )
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_512x512_160k_ade20k/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth'
resume = False

# tta_model = dict(type='SegTTAModel')

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
num_classes = 2

# model settings

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_heads=[1, 2, 5, 8],
        num_layers=[2, 2, 2, 2],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


dataset_type = 'ADE20KDataset'
data_root = '/expand_data/datasets'
img_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

dataset_GF_train = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type=dataset_type,
            metainfo=dict(
                classes=('cloud', 'snow'),
                palette=[[1, 1, 1], [2, 2, 2]]),
            data_root=data_root+'/GFCS/train/',
            reduce_zero_label=True,
            data_prefix=dict(
                img_path='rgb', seg_map_path='label_0_1_2'),
            pipeline=train_pipeline
        ),
        dict(
            type=dataset_type,
            metainfo=dict(
                classes=('cloud', 'snow'),
                palette=[[1, 1, 1], [2, 2, 2]]),
            data_root=data_root+'/GFCS/test/',
            reduce_zero_label=True,
            data_prefix=dict(
                img_path='rgb', seg_map_path='label'),
            pipeline=train_pipeline
        )
    ]
)

dataset_SY_train = dict(
    type='RepeatDataset',
    times=3,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(
            classes=('cloud', 'snow'),
            palette=[[128, 128, 128], [255, 255, 255]]),
        data_root=data_root+'/ZY_cloud/S3_trainset/total/',
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='img', seg_map_path='label'),
        pipeline=train_pipeline
    )
)

train_dataset = dict('ConcatDataset', datasets=[dataset_GF_train, dataset_SY_train])


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataset = dict(
    type=dataset_type,
    metainfo=dict(
        classes=('cloud', 'snow'),
        palette=[[128, 128, 128], [255, 255, 255]]),
    data_root=data_root + '/ZY_cloud/S3_trainset/total/',
    reduce_zero_label=True,
    data_prefix=dict(
        img_path='img', seg_map_path='label'),
    pipeline=test_pipeline
)

batch_size = 32
num_workers = 4
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=train_dataset)
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)
# test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# test_evaluator = val_evaluator

# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [
#                 dict(type='Resize', scale_factor=r, keep_ratio=True)
#                 for r in img_ratios
#             ],
#             [
#                 dict(type='RandomFlip', prob=0., direction='horizontal'),
#                 dict(type='RandomFlip', prob=1., direction='horizontal')
#             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
#         ])
# ]


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

max_epoch = 300
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=20),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=20,
        end=max_epoch,
        by_epoch=True,
    )
]


# training schedule
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epoch, val_interval=1)
val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=20, max_keep_ckpts=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))



