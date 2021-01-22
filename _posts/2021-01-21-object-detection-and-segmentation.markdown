---
layout: post
title:  "object detection and segmentation"
excerpt: "A brief introduction about R-CNN, Fast R-CNN, Faster R-CNN, SSD, YOLO and Mask R-CNN"
date:   2021-01-21 17:00:00
mathjax: true
---

Object detection is a very complex technique that is already widely applied in many fields such as autonomous driving, 
pedestrian detection, surveillance, gesture estimation and so on. It can be split into multi tasks including **object 
localization, object recognition and classification, object segmentation.**

### 1.two-stage object detection

The detection is composed of two separate steps.

#### 1.1.R-CNN

R-CNN (Recursive Convolutional Neural Network) uses **selective search method** to extract 2000 candidate regions which is more likey to 
contain the desired object from the initial image. Then these patches are further processed into the compatible size to overcome the 
input constraints of some CNN which are used for feature extraction (AlexNet, VGG16, VGG19, ResNet, Inception and so on). Finally these 
features is feed into one class-specific linear SVM (support vector machine) to get its classification result and bounding box offset.

<div class="imgcap">
<img src="/assets/detection/rcnn.png">
<div class="thecap">Recursive CNN.</div>
</div>

- selective search method： get region proposals according to the similarity of color, texture
- the labelling of positive and negative samples: the IoU (Intersection over Union) between ground truth bounding box and candidate regions decides if its positive or negative. 
- **non-maximum suppression (NMS)**: when there are multiple prediction bounding boxes for the same object and IoU value between them exceeds the threshold, then we keep the prediction 
bounding box with the maximal prediction probability.

The main disadvantages for R-CNN are:

- the huge computation cost for the selective search method at the initial step.
- Training is a multi-stage pipeline: extracting features, fine-tuning a network with log loss, training SVMs,and finally fitting bounding-box regressors. 
- performs a ConvNet forward pass for each object proposal without sharing computation

#### 1.2.Fast R-CNN

To overcome the drawbacks of R-CNN, the Fast R-CNN is proposed.

<div class="imgcap">
<img src="/assets/detection/fast-rcnn.png">
<div class="thecap">Fast R-CNN.</div>
</div>

- the inputs consist of two parts: the entire image and a set of object proposals.
- it still uses the selective search method which is time-consumed.
- The network first processes the whole image with several convolutional and max pooling layers to produce a conv feature map. Then, for each object proposal a **region of interest (RoI) pooling layer** extracts a fixed-length feature vector from the feature map.
- training is streamlined and single-stage, class and bounding box is trained at the same time using the multi-task loss function.

#### 1.3.Faster R-CNN

It is composed of two modules. The first module is a deep fully convolutional network that proposes regions, and the second module is the Fast R-CNN detector that uses the proposed regions.
It replaces the time-consumed selective search method with using the **region proposal network (RPN)** to get the probable region of interest.

The architecture of Faster R-CNN is demonstrated as follows:
 
<div class="imgcap">
<img src="/assets/detection/faster-rcnn.png">
<div class="thecap">Faster R-CNN.</div>
</div>

A Region Proposal Network (RPN) takes an image as input and outputs a set of rectangular object proposals, each with an objectness score.
We can define some anchor boxes to slide on each point of the feature map. For training RPNs, we assign a binary class label (of being an object or not) to each anchor.

The structure of RPN is depicted as following picture:

<div class="imgcap">
<img src="/assets/detection/RPN.png">
<div class="thecap">Region Proposal Network.</div>
</div>

We further **merge RPN and Fast R-CNN** into a single network by sharing their convolutional features using **alternating training**.

### 2.one-stage object detection

There is only one direct steps for detection..

#### 2.1.SSD


<div class="imgcap">
<img src="/assets/detection/ssd.png">
<div class="thecap">SSD.</div>
</div>

#### 2.2.YOLO


<div class="imgcap">
<img src="/assets/detection/yolo.png">
<div class="thecap">YOLO.</div>
</div>

### 3.object segmentation

### 3.1.Mask R-CNN


<div class="imgcap">
<img src="/assets/detection/mask-rcnn.png">
<div class="thecap">Mask R-CNN.</div>
</div>


### 4.Reference

[Detectron](https://github.com/facebookresearch/Detectron)

[YOLO](https://pjreddie.com/darknet/yolo/)

[Mask R-CNN](https://github.com/matterport/Mask_RCNN)

[tensorflow object detection](https://github.com/tensorflow/models/tree/master/research/object_detection)


