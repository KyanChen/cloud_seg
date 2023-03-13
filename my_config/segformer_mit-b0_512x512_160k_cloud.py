log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='CS',
                 group='CS',
                 name='E20230313_0')
             )
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '../pretrain/segformer_mit-b0_512x512_160k_ade20k_20220617_162207-c00b9603.pth'

resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
num_classes = 3
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
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
        in_channels=[32, 64, 160, 256],
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

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = '/expand_data/datasets'
img_size = (512, 512)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_GF_train1 = dict(
    type=dataset_type,
    data_root=data_root+'/GFCS/train/',
    reduce_zero_label=False,
    img_suffix='.tiff',
    seg_map_suffix='.tif',
    data_prefix=dict(
        img_path='rgb', seg_map_path='label_0_1_2'),
    pipeline=train_pipeline
)

dataset_SY_train = dict(
    type=dataset_type,
    data_root=data_root+'/ZY_cloud/S3_trainset/total/',
    reduce_zero_label=False,
    img_suffix='.jpg',
    seg_map_suffix='.png',
    data_prefix=dict(
        img_path='img', seg_map_path='label'),
    pipeline=train_pipeline
)


dataset_GF_train2 = dict(
    type=dataset_type,
    data_root=data_root+'/GFCS/test/',
    reduce_zero_label=False,
    img_suffix='.tiff',
    seg_map_suffix='.tif',
    data_prefix=dict(
        img_path='rgb', seg_map_path='label'),
    pipeline=train_pipeline
)

train_dataset = dict(type='ConcatDataset', datasets=[dataset_GF_train1, dataset_GF_train1, dataset_GF_train2, dataset_GF_train2, dataset_SY_train])

val_dataset = dict(
    type=dataset_type,
    data_root=data_root + '/ZY_cloud/S3_trainset/total/',
    reduce_zero_label=False,
    img_suffix='.jpg',
    seg_map_suffix='.png',
    test_mode=True,
    data_prefix=dict(
        img_path='img', seg_map_path='label'),
    pipeline=test_pipeline
)


batch_size = 128
num_works = 4
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=num_works,
    train=train_dataset,
    val=val_dataset,
    test=val_dataset)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=20)
evaluation = dict(interval=20, metric='mIoU', pre_eval=True)
