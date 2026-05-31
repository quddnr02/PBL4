# Official SegNeXt-S / MSCAN-S + LightHamHead VOC2012 PBL split config.
# Run from SegNeXt-main with tools/train.py so that the official mmseg registry
# loads mmseg/models/backbones/mscan.py and mmseg/models/decode_heads/ham_head.py.

experiment_name = 'SegNeXt_S_CE_Official'
work_dir = '../runs/SegNeXt_S_CE_Official'

data_root = '../VOCdevkit/VOC2012'
train_split = '../pbl_train.txt'
val_split = '../pbl_val.txt'

num_classes = 21
ignore_index = 255
crop_size = (512, 512)
samples_per_gpu = 4
max_epochs = 30
base_lr = 6e-5
head_lr_mult = 10.0
weight_decay = 0.01
warmup_epochs = 3


def _count_split(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return None


train_count = _count_split(train_split)
val_count = _count_split(val_split)

if train_count is None:
    print(f'[PBL config warning] train split not found yet: {train_split}')
if val_count is None:
    print(f'[PBL config warning] val split not found yet: {val_split}')

print('\n[PBL SegNeXt official experiment]')
print(f'  work_dir                 = {work_dir}')
print('  model                    = official SegNeXt-S (MSCAN-S + LightHamHead)')
print(f'  train split              = {train_split}')
print(f'  val split                = {val_split}')
print(f'  train images count       = {train_count if train_count is not None else "unknown"}')
print(f'  val images count         = {val_count if val_count is not None else "unknown"}')
print(f'  num_classes              = {num_classes}')
print(f'  ignore_index             = {ignore_index}')
print(f'  crop_size                = {crop_size}')
print(f'  optimizer/lr/scheduler   = AdamW lr={base_lr}, head_lr_mult={head_lr_mult}, weight_decay={weight_decay}, linear warmup {warmup_epochs} epochs + poly decay')
print('  best checkpoint          = enabled by evaluation.save_best="mIoU"\n')

norm_cfg = dict(type='BN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/mscan_s.pth'),
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(MD_R=16),
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            ignore_index=ignore_index)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# VOC/PASCAL palette is provided by PascalVOCDataset in MMSegmentation. The split
# files are image-id lists; pbl_val.txt is used only for val/test.
dataset_type = 'PascalVOCDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=ignore_index),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split=train_split,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split=val_split,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split=val_split,
        pipeline=test_pipeline))

optimizer = dict(
    type='AdamW',
    lr=base_lr,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    paramwise_cfg=dict(
        custom_keys={
            'decode_head': dict(lr_mult=head_lr_mult),
            'head': dict(lr_mult=head_lr_mult),
            'norm': dict(decay_mult=0.0),
            'pos_block': dict(decay_mult=0.0),
        }))
optimizer_config = dict()

# SegNeXt's public configs commonly use IterBasedRunner. For PBL bookkeeping this
# config uses EpochBasedRunner with 30 epochs; MMCV's warmup_by_epoch=True makes
# warmup_iters=3 mean three full epochs, then Poly decay runs over max_epochs.
lr_config = dict(
    policy='poly',
    power=1.0,
    min_lr=0.0,
    by_epoch=True,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=warmup_epochs,
    warmup_ratio=1e-6)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=3, save_last=True)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, save_best='mIoU', rule='greater')

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook', by_epoch=True), dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
