# ADE20K Semantic Segmention

## Dataset Preparation
You can download the [ADE20K](https://ade20k.csail.mit.edu/) dataset by running the following commands:
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip -d ADEChallengeData2016
```

The structure of the dataset should be as follows:
```
│ADEChallengeData2016/
├──images/
│  ├──training/
│  │   ├──ADE_train_00000001.jpg
│  │   ├──ADE_train_00000002.jpg
│  │   ├──......
│  ├──validation/
│  │   ├──ADE_val_00000001.jpg
│  │   ├──ADE_val_00000002.jpg
│  │   ├──......
├──annotations/
│  ├──training/
│  │   ├──ADE_train_00000001.png
│  │   ├──ADE_train_00000002.png
│  │   ├──......
│  ├──validation/
│  │   ├──ADE_val_00000001.png
│  │   ├──ADE_val_00000002.png
│  │   ├──......
```


## Getting Started
We provide example scripts in the [scripts](scripts/) folder for training, evaluation.
Note: You may need to modify the dataset path in the [ade20k.py](configs/_base_/datasets/ade20k.py) other configurations (like checkpoint path in the [fpn_enformer_small_ade20k_80k.py](configs/fpn_enformer_small_ade20k_80k.py)) according to your setup.

You can run the following command to start training/evaluation:
```bash
# Training
bash scripts/train.sh

# Evaluation
bash scripts/validate.sh
```


## Acknowledgement
Our semantic segmentation implementation is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [PVT segmentation](https://github.com/whai362/PVT/tree/v2/segmentation). Thank the authors for their wonderful works.
