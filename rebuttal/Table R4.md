
**Q2:**
|Method|Param(M)|FLOPs(G)|4090 Throughput|Top1(%)|
|-|:-:|:-:|:-:|:-:|
|ConvNeur-M1|4.3|0.7|1482.1|75.4|
|ConvNeur-M2|6.1|1.0|1167.9|77.6|
|ConvNeur-M3|10.6|1.8|779.1|80.0|
|ConvNeur-M4|18.1|3.1|522.5|81.5|
|Mamba®-T|9.0|5.1|-|77.4|
|Mamba®-S|28.0|9.9|-|81.1|
|ViG-Ti|7.1|1.3|1543.4|73.9|
|ViG-S|22.7|4.5|731.2|80.4|
|**EnFormer-Small**|**8.1**|**1.1**|**2735.2**|**78.9**|
|**EnFormer-Base**|**14.8**|**2.5**|**1960.9**|**81.2**|
|**EnFormer-Large**|**29.4**|**4.8**|**1201.9**|**82.6**|


**Q3_1:**
|Method|Top-1 Acc.|Aligned soft-JS|
|-|:-:|:-:|
|Partitional+Fuzzy|78.9|0.4485|
|Partitional+Fuzzy+Probabilistic|78.9|0.3573|
|all four base clustering methods|78.6|0.1948|

**Q3_2:**
|Method|Top1(%)|Fuzzy|Probabilistic|Possibilistic|Partitional|
|-|:-:|:-:|:-:|:-:|:-:|
|Partitional+Fuzzy|78.9|0.039|-|-|0.961|
|Partitional+Fuzzy+Probabilistic|78.9|0.148|0.079|-|0.772|
|all four base clustering methods|78.6|0.022|0.108|0.143|0.727|


**Q4**
|Subspace dimensions (Partitional/Fuzzy)|Top1-Acc (%)|
|:-:|:-:|
|16 / 24|78.4|
|24 / 16|78.4|
|20 / 20|78.9|


**Q5**
|Method|Param(M)|FLOPs(G)|Top1-Acc(%)|
|-|:-:|:-:|:-:|
|MoE Clustering|14.6|2.2|80.5|
|EnFormer-Base|14.8|2.5|81.2|
