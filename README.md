#   Lesion  Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing  and CNN-Transformer Fusion Networks  



## Introduction

This is an official release of the paper **Lesion  Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing  and CNN-Transformer Fusion Networks**.



<div align="center" border=> <img src=framework.jpg width="400" > </div>



## Code List

- [x] Network
- [x] Pre-processing
- [x] Training Codes




## Usage

### Dataset

Please download the dataset from [ISIC](https://www.isic-archive.com/) challenge website.

### Pre-processing

Please run:

```bash
$ python src/process_resize.py
$ python src/process_point.py
```

You need to change the **File Path** to your own.



### Training 

### Testing

```bash
$ python test.py --dataset isic2016
```

### Result

|Method | Dice | IoU | Precision | SP | SE |
| :----- | :----: | :----: |:----: |:----: | :----: |
|U-Net | 0.8338 | 0.7486 | 0.9060 | 0.9420 | 0.8330 |
|DeepLabv3+| 0.8610 | 0.7809 | 0.9124 | 0.9438 | 0.8593 |
|SkinFormer| 0.9231 | 0.8551 | **0.9625** | 0.9688 | 0.9008 |
|MPBA-Net| **0.9303** | ***0.8596** | 0.9396 | **0.9692** | **0.9057** |



## Citation

If you find MPBA-Net useful in your research, please consider citing:

```
@inproceedings{
  title={Lesion  Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing  and CNN-Transformer Fusion Networks},
  author={Xuzhen Huang, Yuliang Ma*, Xiajin Mei, ZizhuoWu, Mingxu Sun, Qingshan She}
}
```

