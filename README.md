# Deep Ensemble Clustering for Visual Representation Learning

## Abstract
Recent advances in visual representation learning have seen the rise of clustering-based vision backbones, which adopt clustering as a core paradigm for feature extraction. However, existing clustering-based backbones typically rely on a single clustering algorithm, whose inherent inductive bias limits their representational capacity. To address this, we propose EnFormer, which embeds ensemble clustering as a core component of feature extraction. EnFormer structures feature extraction around two steps: (i) Ensemble Generation, where several differentiable base clustering methods are introduced to capture diverse semantic structures; and (ii) Consensus Aggregation, which employs a differentiable mechanism to fuse the results of all base clusterings to reconstruct refined visual features. Extensive experiments show that EnFormer consistently outperforms existing clustering-based backbones across core vision tasks, with higher performance and significantly improved throughput.


## Results and Pre-trained/Fine-tuned Models

### ImageNet-1K pre-trained models for classification
|    name    | resolution | epochs | acc@1 | #params | FLOPs | throughput  |       model       |
|:----------:|:----------:|:------:|:-----:|:-------:|:-----:|:-----------:|:-----------------:|
| EnFormer-S |  224x224   |  310   | 78.9  |  8.1M   | 1.1G  | 1476.5img/s |    Coming Soon    |
| EnFormer-B |  224x224   |  310   | 81.2  |  14.8M  | 2.5G  | 1075.5img/s |    Coming Soon    |
| EnFormer-L |  224x224   |  310   | 82.6  |  29.4M  | 4.8G  | 621.7img/s  |    Coming Soon    |

### COCO2017 fine-tuned models for object detection and instance segmentation
|    name     |   Method   | Lr Schd | box mAP | mask mAP | #params | Fine-tuned Model  |
|:-----------:|:----------:|:-------:|:-------:|:--------:|:-------:|:-----------------:|
| EnFormer-S  | Mask R-CNN |   1x    |  41.3   |   38.1   |  28.2M  |    Coming Soon    |
| EnFormer-B  | Mask R-CNN |   1x    |  42.8   |   39.3   |  35.0M  |    Coming Soon    |
| EnFormer-L  | Mask R-CNN |   1x    |  44.0   |   40.0   |  50.3M  |    Coming Soon    |

### ADE20K fine-tuned models for semantic segmentation
|    name    |    Method    | Lr Schd | mIoU | #params | Fine-tuned Model  |
|:----------:|:------------:|:-------:|:----:|:-------:|:-----------------:|
| EnFormer-S | Semantic FPN |   80k   | 43.3 |  12.3M  |    Coming Soon    |
| EnFormer-B | Semantic FPN |   80k   | 44.3 |  19.3M  |    Coming Soon    |
| EnFormer-L | Semantic FPN |   80k   | 46.6 |  34.6M  |    Coming Soon    |


## Installation
To set up the environment, run the following commands:
```conda
conda create -n enformer python=3.9 
conda activate enformer
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


## Getting Started

### Image Classification
Please refer to [classification/README.md](classification/README.md) for details.

### Object Detection and Instance Segmentation
Please refer to [detection/README.md](detection/README.md) for details.

### Semantic Segmentation
Please refer to [segmentation/README.md](segmentation/README.md) for details.


## LICENSE
This project is released under the [Apache 2.0 license](LICENSE).


[//]: # (## Citation)

[//]: # (To be continued...)


