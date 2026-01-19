# COCO2017 Object Detection and Instance Segmentation

## Dataset Preparation
You can download the [COCO](https://cocodataset.org/)2017 dataset by running the following commands:
```bash
# Download and Unzip training data
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Download and Unzip validation data
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Download and Unzip annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

The structure of the dataset should be as follows:
```
│coco/
├──train2017/
│  ├──000000000009.jpg
│  ├──000000000025.jpg
│  ├──......
├──val2017/
│  ├──000000000139.jpg
│  ├──000000000285.jpg
│  ├──......
├──annotations/
│  ├──instances_train2017.json
│  ├──instances_val2017.json
│  ├──......
```


## Getting Started
We provide example scripts in the [scripts](scripts/) folder for training, evaluation.
Note: You may need to modify the dataset path in the [coco_instance.py](configs/_base_/datasets/coco_instance.py) other configurations (like checkpoint path in the [mask_rcnn_enformer_small_fpn_1x_coco.py](configs/mask_rcnn_enformer_small_fpn_1x_coco.py)) according to your setup.

You can run the following command to start training/evaluation:
```bash
# Training
bash scripts/train.sh

# Evaluation
bash scripts/validate.sh
```


## Acknowledgement

Our detection implementation is based on [MMDetection](https://github.com/open-mmlab/mmdetection/) and [PVT detection](https://github.com/whai362/PVT/tree/v2/detection). Thank the authors for their wonderful works.
