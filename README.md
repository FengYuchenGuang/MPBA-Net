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
$ python src/BinaryMapResize.py
$ python src/BinaryGroundPatchMap.py
```

You need to change the **File Path** to your own.



### Training 

### Testing

```bash
$ python test.py --dataset isic2018
```

### Result

|Method | Dice | IoU | Precision | SP | SE |
| :----- | :----: | :----: |:----: |:----: | :----: |
|U-Net | 0.8232 | 0.7274 | 0.9268 | 0.9546 | 0.7868 |
|DeepLabv3+| 0.8494 | 0.7612 | 0.9245 | 0.9544 | 0.8266 |
|SkinFormer| 0.8868 | 0.8184 | **0.9483** | 0.9562 | 0.8890 |
|MPBA-Net| **0.8893** | **0.8205** | 0.9260 | **0.9576** | **0.9026** |



## Citation

If you find MPBA-Net useful in your research, please consider citing:

```
@inproceedings{
  title={Lesion  Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing  and CNN-Transformer Fusion Networks},
  author={Xuzhen Huang, Yuliang Ma*, Xiajin Mei, ZizhuoWu, Mingxu Sun, Qingshan She}
}
```

