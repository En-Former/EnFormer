_base_ = ['./_base_/default_runtime.py', './_base_/models/fpn_r50.py', './_base_/datasets/ade20k.py']

work_dir = f'./work_dirs/fpn_enformer_large_ade20k_80k'
arch = 'L'
neck_channels = {'S': [40, 80, 160, 320], 'B': [64, 128, 320, 512], 'L': [64, 128, 320, 512]}
checkpoint = "enformer_large.pth.tar"  # noqa

# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='enformer_seg',
        arch=arch,
        task='segmentation',
        out_indices=(0, 1, 2, 3),
        drop_path_rate=(0., 0.),
        with_positional_encoding=True,
        witch_cp=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    neck=dict(in_channels=neck_channels[arch]),
    decode_head=dict(num_classes=150))

gpu_multiples = 1
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')
device = 'cuda'
