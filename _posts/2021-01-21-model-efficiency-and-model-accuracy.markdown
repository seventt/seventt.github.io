---
layout: post
title:  "MobileNets, EfficientNet and EfficientDet"
excerpt: "suitable network architecture design to achieve the trade-off between model efficiency (computing and storage resources, model size and latency) and model accuracy"
date:   2021-01-21 17:00:00
mathjax: true
---

### 1.MobileNetV1

MobileNet replaces standard convolution with Depth-wise separable convolution and point-wise convolution.

<div class="imgcap">
<img src="/assets/mobilenets/depthwise-conv.png">
<div class="thecap">Depth-wise separable convolution and point-wise convolution.</div>
</div>

computation cost for standard convolution:

$$
\begin{equation}
D_K \times D_K \times M \times N \times D_F \times D_F \\
D_F: \text{input feature map size} \\
M: \text{input channels} \\
D_M: \text{filter kernel size} \\
N: \text{filter number or output channels}
\end{equation}
$$

computation cost for depth-wise separable convolution and \\( 1 \times 1 \\) point-wise convolution, **the second term takes the bigger part of computation cost**.

$$
\begin{equation}
D_K \times D_K \times M \times D_F \times D_F + M \times N \times D_F \times D_F \\
\text{first term is for depth-wise separable convolution} \\
\text{second term is for point-wise convolution} 
\end{equation}
$$

The role of the **width multiplier** \\( \alpha \\) is to thin a network uniformly at each layer. The computation cost for these thinner models:

$$
\begin{equation}
D_K \times D_K \times \alpha M \times D_F \times D_F + \alpha M \times \alpha N \times D_F \times D_F 
\end{equation}
$$

Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly \\( \alpha^2 \\)

Another hyper-parameter to reduce the computational cost of a neural network is a **resolution multiplier** \\( p \\),  
we apply this to the input image and the internal representation of every layer is subsequently reduced by the same multiplier

$$
\begin{equation}
D_K \times D_K \times \alpha M \times p D_F \times p D_F + \alpha M \times \alpha N \times p D_F \times p D_F 
\end{equation}
$$

Resolution multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly \\( p^2 \\)

MobileNet structure:

<div class="imgcap">
<img src="/assets/mobilenets/mobilenet.png">
<div class="thecap">MobileNet body architecture.</div>
</div>

### 2.ShuffleNet

using point-wise group convolution and channel shuffle to replace previous \\( 1 \times 1 \\) point-wise convolution in order to further reduce the computation cost.

The following figure illustrates why the operation of uniform channel shuffle is important. It ensures the communication between different group features after the group convolution.

<div class="imgcap">
<img src="/assets/mobilenets/channel-shuffle.png">
<div class="thecap">channel shuffle.</div>
</div>

The ShuffleNet unit.

<div class="imgcap">
<img src="/assets/mobilenets/shuffleunit.png">
<div class="thecap">ShuffleNet unit.</div>
</div>

### 3.MobileNetV2

the main contribution is a novel layer module : the inverted residual with linear bottleneck. 
It takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. 
Features are subsequently projected back to a low-dimensional representation with a linear convolution.
Another change is to remove non-linearities in the narrow layers to maintain representational power.

inverted residual block: the intermediate layers are thicker. By reducing the number of input channels the computation cost could be reduced.
<div class="imgcap">
<img src="/assets/mobilenets/inverted-residual-block.png">
<div class="thecap">inverted residual block.</div>
</div>

traditional residual block: the intermediate layers are thinner.

<div class="imgcap">
<img src="/assets/mobilenets/residual-block.png">
<div class="thecap">residual block.</div>
</div>

### 4.SENet

SENet(Squeeze-Excitation-Net): explicitly model the interdependencies between the channels of convolutional features
**squeeze** operation: Each of the learned filters operates with a local receptive field and consequently each unit of the transformation output U is unable to exploit contextual information **outside** of this region,
To mitigate this problem, we propose to squeeze global spatial information into a channel descriptor. This is achieved by using global average pooling to generate channel-wise statistics.
**excitation** operation is to ensure that multiple channels are allowed to be emphasised (rather than enforcing a one-hot activation). To meet these criteria, we opt to employ a simple gating mechanism with a sigmoid activation:

<div class="imgcap">
<img src="/assets/mobilenets/senet.png">
<div class="thecap">Illustration of Squeeze and Excitation Operation.</div>
</div>

The schema of combination between SEBlock and Inception/Residual model is described as following:

<div class="imgcap">
<img src="/assets/mobilenets/senet-block.png">
<div class="thecap">Squeeze and Excitation Block.</div>
</div>

### 5.MnasNet

It uses automated neural architecture search (NAS) approach for designing mobile models by reinforcement learning.
we explicitly incorporate the speed information into the main reward function of the search algorithm, so that the search can identify a model that achieves a good trade-off between accuracy and speed

The overall flow of our approach consists mainly of three components: a RNN-based controller for learning and sampling model architectures, 
a trainer that builds and trains models to obtain the accuracy, and an inference engine for measuring the model speed on real mobile phones using TensorFlow Lite.

<div class="imgcap">
<img src="/assets/mobilenets/NAS-for-mobile.png">
<div class="thecap">Neural architecture search for mobile.</div>
</div>

MnasNet architecture:

<div class="imgcap">
<img src="/assets/mobilenets/MnasNet.png">
<div class="thecap">MnasNet architecture.</div>
</div>

Factorized Hierarchical Search Space: It introduces a novel
factorized hierarchical search space that factorizes a CNN
model into unique blocks and then searches for the operations and connections per block separately, thus allowing
different layer architectures in different blocks. Our intuition is that we need to search for the best operations based
on the input and output shapes to obtain better accuratelatency trade-offs.

<div class="imgcap">
<img src="/assets/mobilenets/factorized-hierarchical-search-space.png">
<div class="thecap">Factorized Hierarchical Search Space.</div>
</div>

### 6.EfficientNet

simultaneous compound scaling on width (filter number), depth (layer number), resolution (image size) to find the model architecture with the best
efficiency and accuracy.

<div class="imgcap">
<img src="/assets/mobilenets/compound-scale.png">
<div class="thecap">compound scale on width, depth and resolution.</div>
</div>

EfficientNet-B0 architecture is similar to MnasNet but a little bigger. Its main building block is mobile inverted bottleneck MBConv。

<div class="imgcap">
<img src="/assets/mobilenets/EfficientNet-B0.png">
<div class="thecap">EfficientNet-B0 architecture.</div>
</div>

EfficientNet-B0 to B7 is acquired as follows:
- step 1: we first fix \\( \phi = 1 )\\, assuming twice more resources available, and do a small grid search of \\(\alpha, \beta, \gamma)\\ based on the following equation.
 In particular, we find the best values for EfficientNet-B0 are \\(\alpha = 1.2, \beta = 1.1,\gamma=1.15)\\.
 
$$
\begin{equation}
depth: d = \alpha^\phi \\
width: w = \beta^\phi \\
resolution: r = \gamma^\phi \\
\quad \quad subject to: \alpha \times \beta^2 \times \gamma^2 = 2
\end{equation}
$$

- step 2: we then fix \\(\alpha, \beta, \gamma)\\ as constants and scale up baseline network with different \\(\phi)\\, to obtain EfficientNet-B1 to B7

EfficientNet performance on ImageNet:

<div class="imgcap">
<img src="/assets/mobilenets/efficientnet-performance.png">
<div class="thecap">EfficientNet performance on ImageNet.</div>
</div>

compound scaling for MobileNets and ResNet

<div class="imgcap">
<img src="/assets/mobilenets/scaling-mobilenets-and-resnet.png">
<div class="thecap">compound scaling for MobileNets and ResNet.</div>
</div>

### 7.EfficientDet

Firstly, it incorporates weighted bi-directional feature pyramid network (BiFPN) rather than simple feature pyramid network (FPN), which allows easy and fast multi-scale feature fusion; 
Secondly, it uses a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks.
This mainly follows the one-stage detector design, and aims to achieve both better efficiency and higher accuracy with optimized network architectures.

Backbone network: reuse the same width/depth scaling coefficients of EfficientNet-B0 to B6 such that we can easily reuse their ImageNet-pretrained checkpoints

BiFPN structure:

<div class="imgcap">
<img src="/assets/mobilenets/BiFPN.png">
<div class="thecap">Feature Network Design.</div>
</div>

compound scaling:

- BiFPN:

$$
\begin{equation}
W_{bifpn} = 64 \times 1.35 ^ \phi \quad D_{bifpn} = 3 + \phi
\end{equation}
$$

- Box/class prediction network:

$$
\begin{equation}
W_pred = W_{bifpn} \quad  D_{box} = D_{class} = 3 + floor(\frac{\phi}{3})
\end{equation}
$$

- Input image resolution:

$$
\begin{equation}
R_input = 512 + \phi * 128
\end{equation}
$$

- compound scaling results for EfficientDet D0-D6

<div class="imgcap">
<img src="/assets/mobilenets/EfficientDets.png">
<div class="thecap">compound scaling results for EfficientDet D0-D6.</div>
</div>

EfficientDet architecture:

<div class="imgcap">
<img src="/assets/mobilenets/EfficientDet.png">
<div class="thecap">EfficientDet architecture.</div>
</div>

### 4.Reference

[ShuffleNet](https://zhuanlan.zhihu.com/p/32304419)

[MnasNet](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html)

[EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)

[EfficientDet](https://github.com/google/automl/tree/master/efficientdet)

[FPN](https://zhuanlan.zhihu.com/p/62604038)





