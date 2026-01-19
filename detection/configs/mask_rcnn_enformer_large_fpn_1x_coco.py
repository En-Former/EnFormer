_base_ = ['_base_/default_runtime.py', '_base_/datasets/coco_instance.py',
          '_base_/schedules/schedule_1x.py', '_base_/models/mask_rcnn_r50_fpn.py']

work_dir = f'./work_dirs/mask_rcnn_enformer_large_fpn_1x_coco'
arch = 'L'
neck_channels = {'S': [40, 80, 200, 320], 'B': [64, 128, 320, 512], 'L': [64, 128, 320, 512]}
checkpoint = "enformer_large.pth.tar"  # noqa

# optimizer
model = dict(
    backbone=dict(
        type='enformer_det',
        arch=arch,
        task='detection',
        out_indices=(0, 1, 2, 3),
        drop_path_rate=(0., 0.),
        with_positional_encoding=True,
        witch_cp=False,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=neck_channels[arch],
        out_channels=256,
        num_outs=5))

find_unused_parameters=True
