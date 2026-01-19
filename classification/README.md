# ImageNet-1K Classification

## Dataset Preparation

You can download the [ImageNet-1K](http://image-net.org/) dataset by running the following commands:
```bash
# Download training data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate

# Download validation data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate

# Download the development kit
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
```

Then you can extract by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4) and get the following folder structure:
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## Getting Started
We provide example scripts in the [scripts](scripts/) folder for training, evaluation, and visualization.
Note: You may need to modify the dataset path/image path and other configurations in the script according to your setup.

You can run the following command to start training/evaluation/visualization:
```bash
# Training
bash scripts/train.sh

# Evaluation
bash scripts/eval.sh

# Visualization
bash scripts/visualize.sh
```
