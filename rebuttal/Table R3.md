**Q1:**
|Method|Param(M)|FLOPs(G)|A40 Throughput|Top1(%)|
|-|:-:|:-:|:-:|:-:|
|ResNet18|12.0|1.8|5152.7|69.8|
|ResNet50|26.0|4.1|1528.6|79.8|
|ConvMixer-512/16|5.4|4.4|520.5|73.8|
|ConvMixer-1024/12|14.6|13.1|311.3|77.8|
|ConvMixer-768/32|21.1|5.0|795.5|80.2|
|ViT-B/16|86.0|55.5|383.9|77.9|
|ViT-L/16|307.0|190.7|116.5|76.5|
|PVT-Tiny|13.2|1.9|1824.2|75.1|
|PVT-Small|24.5|3.8|1001.7|79.8|
|Swin-Tiny|29.0|4.5|833.0|81.3|
|Swin-Small|50.0|8.7|485.2|83.0|
|ResMLP-12|15.0|3.0|1835.8|76.6|
|ResMLP-24|30.0|6.0|912.2|79.4|
|ResMLP-36|45.0|8.9|602.2|79.7|
|MLP-Mixer-B/16|59.0|12.7|544.5|76.4|
|MLP-Mixer-L/16|207.0|44.8|161.2|71.8|
|gMLP-Ti|6.0|1.4|2280.9|72.3|
|gMLP-S|20.0|4.5|956.3|79.6|
|CoC-Tiny|5.3|1.1|953.8|71.8|
|CoC-Small|14.0|2.8|808.8|77.5|
|CoC-Medium|27.9|5.9|328.4|81.0|
|FEC-Small|5.5|1.4|935.0|72.7|
|FEC-Base|14.4|3.4|768.1|78.1|
|FEC-Large|28.3|6.5|352.1|81.2|
|**EnFormer-Small**|**8.1**|**1.1**|**1465.2**|**78.9**|
|**EnFormer-Base**|**14.8**|**2.5**|**1098.9**|**81.2**|
|**EnFormer-Large**|**29.4**|**4.8**|**645.4**|**82.6**|

**Q6:**
|EnsembleComponents|Param(M)|Throughput|Top1(%)|TrainingMem(GB)|InferenceMem(GB)|
|-|:-:|:-:|:-:|:-:|:-:|
|Baseline (w/o ensemble)|7.6|1745.7|77.0|48.00|5.58|
|Partitional|7.8|1533.1|77.6|51.54|6.16|
|Fuzzy|7.8|1566.3|78.0|50.76|6.16|
|Partitional+Fuzzy|8.1|1476.5|78.9|57.51|6.16|
|Partitional+Possibilistic|8.1|1466.8|78.6|55.92|6.16|
|Partitional+Probabilistic|8.1|1377.2|78.8|70.38|6.16|
|Partitional+Fuzzy+Possibilistic|8.1|1450.7|78.5|57.74|6.16|
|Partitional+Fuzzy+Probabilistic|8.1|1332.0|78.9|68.85|6.16|
|all four base clustering methods|8.3|1306.8|78.6|69.56|6.16|

