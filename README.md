### pytorch Implementation of U-Net, R2U-Net, Attention U-Net, Attention R2U-Net, TransU-Net

**(This repository is forked from )**

**U-Net: Convolutional Networks for Biomedical Image Segmentation**

https://arxiv.org/abs/1505.04597

**Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation**

https://arxiv.org/abs/1802.06955

**Attention U-Net: Learning Where to Look for the Pancreas**

https://arxiv.org/abs/1804.03999

**Attention R2U-Net : Just integration of two recent advanced works (R2U-Net + Attention U-Net)**

## U-Net
![U-Net](img/U-Net.png)


## R2U-Net
![R2U-Net](img/R2U-Net.png)

## Attention U-Net
![AttU-Net](img/AttU-Net.png)

## Attention R2U-Net
![AttR2U-Net](img/AttR2U-Net.png)

## TransU-Net

Before use, we need first to download pretrain model.
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
All the supported models:
- ViT-B_16
- ViT-B_32
- ViT-L_16
- R50+ViT-B_16
- R50+ViT-L_16


```shell
# This script will automatically download the pretrained models to the folder ./pretrain/imagenet21k
run_scripts/download_pretrained_models.sh
```



## Evaluation
We just test the models with [ISIC 2018 dataset](https://challenge.isic-archive.com/data/#2018) task 1. The dataset was split into three subsets, training set, validation set, and test set, which the proportion is 70%, 10% and 20% of the whole dataset, respectively. The train dataset contains 2594 images, the validation dataset contains 100 images, the test dataset contains 1000 images.

![evaluation](img/Evaluation.png)
