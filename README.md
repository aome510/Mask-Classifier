# Mask Classifier

## Introduction

This repository contains codes for training and evaluating the Mask Classifier model, which is mostly implemented in ***Python 3*** and ***Keras*** framework.

When training and testing the model, we mostly use [SSH](https://github.com/mahyarnajibi/SSH) to detect and crop face images.

Please read our paper at [here](./paper/paper.pdf) for further details on preparing the datasets as well as training and testing the model.

This project was supported by [VinAI research](https://www.vinai.io).
## People

Our team consists of four members including me (***Thang Pham***), ***Bao Tran***, ***Duy Pham***, and ***Long Nguyen***.

This project was performed while the authors interned at [VinAI research](https://www.vinai.io/).

## Model Performance

Our team has tried different network architectures and trained on different frameworks such as ***PyTorch***, ***Keras***, and ***Caffe***.

We observed that deploying ***ResNet50*** pretrained network brought us high accuracy model as well as fast training time.

During the experiment, the Mask Classifer model can achieve about ***96.5%*** accuracy and a performance of ***10-15 FPS*** on a Nvidia's GeForce GTX Titan X machine.

### Some of the results
![](./data/demo/output/00001.jpg)
![](./data/demo/output/00002.jpg)
![](./data/demo/output/00003.jpg)
![](./data/demo/output/00004.jpg)

## Contents

1. [Installation](#install)
2. [Running the demo](#demo)
3. [Training a model](#train)

<a name="install"></a>
## Installation
1. Clone the repository:
```
git clone https://github.com/aome510/Mask-Classifier.git
```
2. Setup SSH:
* [Install and setup SSH requirements](./lib/SSH/README.md#installation)
* [Install SSH caffemodel](./lib/SSH/README.md#demo)
3. Install python requirements:
```
pip install -r requirements.txt
```

<a name="demo"></a>
## Running the demo
To run the demo, first you need to download the pre-trained mask classifier model from [here](https://drive.google.com/file/d/1Ogew-mbiMqznzO3ok_T-IiqQGpq4OHw5) and then save it into a folder named ```model/```.

After downloading the model, you can run demo on webcam:
```
python demo.py
```
If you want to demo on videos, you can download our demo videos from [here](https://drive.google.com/drive/folders/1PSw1eDErcjLaOkIUxYXbLEacBNdaMjHQ) and then save it into a folder named ```data/videos/```. After that, you can modify ```demo.py``` to run demo on videos.

<a name="train"></a>
## Training the model

### Preparing the datasets

1. ***CelebA*** dataset:

    You can download ***CelebA*** dataset from [here](https://www.kaggle.com/jessicali9530/celeba-dataset) and then save it into a folder named ```data/celebA/```

2. ***WiderFace*** dataset:

    You can download ***WiderFace*** dataset from [here](https://drive.google.com/open?id=16a0TSvPI_3up7tbblxTHxXj5hZ9N7RyH) and then save it into a folder named ```data/WiderFace/```
3. ***MAFA*** dataset:

    You can download ***MAFA*** dataset from [here](https://drive.google.com/drive/folders/1nbtM1n0--iZ3VVbNGhocxbnBGhMau_OG) and then save it into a folder named ```data/MAFA/```
4. ***Mask Classifier*** dataset (our dataset):

    You can download ***Mask Classifier*** dataset from [here](https://drive.google.com/open?id=1WLKCLvR_nXSsOUywqKixoetz03WoJ8cL) and then save it into a folder name ```data/mask_classifier/```

### Combining the datasets
After downloading all the datasets listed above you can run:
```
python gen_data.py
```
to combine the datasets and split them for training and cross-validating the model.

### Training the model

After preparing the combined dataset, you can start training the model:
```
python train.py
```

By default, we use ***resnet50*** network to train the model. You can change to ***reception_resnet_v2*** network by modifying ```train.py```.

Please read [this](https://keras.io/applications/) if you want to train with diffrent network architectures.
