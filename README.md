The official implementation of <a href="https://arxiv.org/pdf/2407.17083">When Text and Images Don't Mix: Bias-Correcting Language-Image Similarity Scores for Anomaly Detection</a> 
by Adam Goodge, Bryan Hooi and Wee Siong Ng and to appear in The 35th British Machine Vision Conference (BMVC2024).

The code is organized as follows:

## Files
- ```clip``` : contains files for instantiating the CLIP model. Please download the model weights first.
- ```imagenet_utils``` : contains files and utility functions for ImageNet data.
- ```data_utils.py``` : contains functions for downloading and instantiating data.
- ```cifar_eval.py``` : the main file for cifar10 and cifar100 experiments.
- ```imagenet_eval.py``` :  the main file for tinyimagnet experiments.

## Citation
```
**@inproceedings{goodge2022lunar,
  title={LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks},
  author={Goodge, Adam and Hooi, Bryan and Ng, See Kiong and Ng, Wee Siong},
  journal={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022}
}**
```
