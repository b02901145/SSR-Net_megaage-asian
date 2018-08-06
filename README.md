# SSR-Net_megaage-asian
[IJCAI18] SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation 
  * A real time age estimation model with 0.32MB
  * This repository is for *MegaAge-Asian datasets*
  * See also https://github.com/shamangary/SSR-Net for more datasets and video demo

**Last update: 2018/08/06 (Adding MegaAge-Asian dataset.)**

<img src="https://media.giphy.com/media/ygBDe4FIU4Cybbfh2N/giphy.gif" height="240"/> <img src="https://media.giphy.com/media/bZvHMOp2hBsusr96fa/giphy.gif" height="240"/> 

<img src="https://github.com/shamangary/SSR-Net/blob/master/demo/TGOP_tvbs.png" height="240"/> <img src="https://github.com/shamangary/SSR-Net/blob/master/demo/the_flash_cast.png" height="240"/>

<img src="https://github.com/shamangary/SSR-Net/blob/master/table1.png" height="240"/>

<img src="https://github.com/b02901145/SSR-Net_megaage-asian/blob/master/paper_images/magaage_asian_CA.png" height=120>

## Paper
### PDF
https://github.com/shamangary/SSR-Net/blob/master/ijcai18_ssrnet_pdfa_2b.pdf
### Authors
**[Tsun-Yi Yang](http://shamangary.logdown.com/), [Yi-Husan Huang](https://github.com/b02901145), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_zh.html), [Pi-Cheng Hsiu](https://www.citi.sinica.edu.tw/pages/pchsiu/index_en.html), and [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)**

## Abstract
This paper presents a novel CNN model called Soft Stagewise Regression Network (SSR-Net) for age estimation from a single image with a compact model size. Inspired by DEX, we address age estimation by performing multi-class classification and then turning classification results into regression by calculating the expected values. SSR-Net takes a coarse-to-fine strategy and performs multi-class classification with multiple stages. Each stage is only responsible for refining the decision of the previous stage. Thus, each stage performs a task with few classes and requires few neurons, greatly reducing the model size. For addressing the quantization issue introduced by grouping ages into classes, SSR-Net assigns a dynamic range to each age class by allowing it to be shifted and scaled according to the input face image. Both the multi-stage strategy and the dynamic range are incorporated into the formulation of soft stagewise regression. A novel network architecture is proposed for carrying out soft stagewise regression. The resultant SSR-Net model is very compact and takes only **0.32 MB**. Despite of its compact size, SSR-Net’s performance approaches those of the state-of-the-art methods whose model sizes are more than 1500x larger.

## Platform
+ Keras
+ Tensorflow
+ GTX-1080Ti and GTX-1080
+ Ubuntu

## Codes
**This repository is for MegaAge-Asian datasets.**
There are three different section of this project.
+ Data pre-processing
+ Training
+ Testing

We will go through the details in the following sections.


### Data pre-processing
+ Download MegaAge-Asian dataset from http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/
+ Extract under this folder
+ Run the following codes for dataset pre-processing 

   or  

+ Download from https://drive.google.com/open?id=1CismL8x4gi3sAfTi3qpxedWSStTPsrcp
```
python TYY_Megaage_asian_create_db.py
```

### Training
+ For SSR-Net
```
bash run_ssrnet_megaage.sh
```
+ For MobileNet
```
bash run_megaage_MobileNet.sh
```
+ For DenseNet
```
bash run_megaage_DenseNet.sh
```

### Testing
Create predicted results and calculate CA (cumulative accuracy)
+ For SSR-Net, MobileNet and DenseNet
```
bash run_CA.sh
```
