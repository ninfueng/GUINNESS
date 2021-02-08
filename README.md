# GUINNESS: A GUI based binarized Neural NEtwork SyntheSizer toward an FPGA (Supporting PyQt5 version)
**This fork was created to make GUINNESS compatible with PyQt5.**
Some un-used dependencies were removed. The additional downloading script for a dataset mentioned in `guinness_tutorial1_v2.pdf` is also included. With PyQt5 supporting, this allows the user to not build PyQt4 from binary.

- To download a dataset mentioned in `guinness_tutorial1_v2.pdf` and unzip.
```bash
bash download_dataset.sh
```
- To train or to use the GUINNESS please follows `guinness_tutorial1_v2.pdf`.

## TODO: 
- [ ] Qcursor fixing.

## -------------------------------------------------------------------------------------------
This GUI based framework includes both a training on a GPU, and a bitstream generation for an FPGA using the Xilinx Inc. SDSoC. This tool uses the Chainer deep learning framework to train a binarized CNN. Also, it uses optimization techniques for an FPGA implementation. Details are shown in following papers:

[Nakahara IPDPSW2017] H. Yonekawa and H. Nakahara, "On-Chip Memory Based Binarized Convolutional Deep Neural Network Applying Batch Normalization Free Technique on an FPGA," IPDPS Workshops, 2017, pp. 98-105.  

[Nakahara FPL2017] H. Nakahara et al., "A Fully Connected Layer Elimination for a Binarized Convolutional Neural Network on an FPGA", FPL, 2017, pp. 1-4.

[Nakahara FPL2017 Demo] H. Nakahara et al., "A demonstration of the GUINNESS: A GUI based neural NEtwork SyntheSizer for an FPGA", FPL, 2017, page 1.

### 1. Requirements:

Ubuntu 16.04 LTS (14.04 LTS is also supported)  

Python 3.5.1
(Note that, my recommendation is to install by Anaconda 4.1.0 (64bit)+Pyenv,
 for Japanese Only, I prepared the Python 3.5 by following http://blog.algolab.jp/post/2016/08/21/pyenv-anaconda-ubuntu/)

CUDA 8.0 (+GPU), CuDNN 6.0
(Also, you must sign up the NVidia developer account)

Chainer 1.24.0 + CuPy 2.0

Xilinx Inc. SDSoC 2017.4

FPGA board: Xilinx ZC702, ZC706, ZCU102, Digilent Zedboard, Zybo  
(Soon, I will support Intel's FPGAs!, and the PYNQ board)  

<s> PyQt4 </s> PyQt5, matplotlib, OpenCV3, numpy, <s> scipy </s>,
(Above libraries are installed by the Anaconda, however, you must individually install the OpenCV by "conda install -y -c menpo opencv3")

### 2. Setup Libraries

 Install the following python libraries:

 Chainer 
```
pip install chainer==1.24
 ```
 <s>  PyQt4 (not PyQt5!), it is already installed by the Anaconda </s> PyQt5
```bash
conda install pyqt
 ```

 OpenCV3
```bash
conda install -y -c menpo opencv3
```
### 3. Run GUINNESS
```bash
python guinness.py
```
### 4. Tutorial

 Read a following document (25/Oct./2017 Updated!!)

 1 The GUINNESS introduction and BCNN implementation on an FPGA  
 guinness_tutorial1_v2.pdf <https://www.dropbox.com/s/oe6gptgyi4y92el/guinness_tutorial1_v2.pdf?dl=0>

 2 The GUINNESS for the Intel FPGAs (Soon, will be uploaded)
 
 3 Pedestrian detection (Under preparing)

 4 Make a custom IP core for your own FPGA board (Under preparing) 

### 5. On-going works
 This is a just trial version. I have already developed the extend version including following ones.
 
 Supporing the Intel's FPGA (DE5-net, DE10-nano, and DE5a-net boards with the Intel SDK for OpenCL)
 
 High performance image recognition (fully pipelined and SIMD CNNs)  
 
 Object detector on a low-cost FPGA (e.g., pedestrian detection)

FPGA YOLOv2 (ZCU102 board)

[![FPGA YOLOv2 ON YOUTUBE](http://img.youtube.com/vi/_iMboyu8iWc/0.jpg)](https://www.youtube.com/watch?v=_iMboyu8iWc&t=5s)

Pedestrian Detector (Zedboard)

[![Pedestrian Detector ON YOUTUBE](http://img.youtube.com/vi/X82PVBuAuuo/0.jpg)](https://www.youtube.com/watch?v=X82PVBuAuuo&list=FLIIfj2LoI2TVWF5wQkZHiHg)


 If you are interesting the extended one, please, contact me.

### 6. Acknowledgements
 This work is based on following projects:

 Chainer binarized neural network by Daisuke Okanohara  
 https://github.com/hillbig/binary_net

 Various CNN models including Deep Residual Networks (ResNet)   
  for CIFAR10 with Chainer by mitmul  
 https://github.com/mitmul/chainer-cifar10

 This research is supported in part by the Grants in Aid for Scientistic Research of JSPS,  
and an Accelerated Innovation Research Initiative Turning Top Science and Ideas into High-Impact  
Values program(ACCEL) of JST. Also, thanks to the Xilinx University Program (XUP), Intel University Program,
 and the NVidia Corp.'s support.
