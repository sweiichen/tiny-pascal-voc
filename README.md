# Tiny Pascal VOC 

HW3 for NCTU CS Selected Topics in Visual Recognition using Deep Learning
Tiny VOC dataset contains only 1,349 training images.This assignment is to implement the instance segmentation task.
    I reference from the [pytorch offical tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) and use ImageNet(resnet152) pre-trained backbone and reference the training method of Mask R-CNN from the tutoiral to do instance segmentation learning.
Moreover, I didn't use any COCO or original Pascal VOC dataset pretrained model on this assignment.
The used python code, [utils.py](https://github.com/sweiichen/tiny-pascal-voc/blob/main/utils.py), [transforms.py](https://github.com/sweiichen/tiny-pascal-voc/blob/main/transforms.py), [coco_eval.py](https://github.com/sweiichen/tiny-pascal-voc/blob/main/coco_eval.py), [engine.py](https://github.com/sweiichen/tiny-pascal-voc/blob/main/engine.py) and [coco_utils.py](https://github.com/sweiichen/tiny-pascal-voc/blob/main/coco_utils.py), are from https://github.com/pytorch/vision.git.

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- GeForce GTX 1080 Ti

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Inference](#inference)

## Installation
All requirements should be detailed in [requirements.txt](https://github.com/sweiichen/tiny-pascal-voc/blob/main/requirements.txt). 
You have to create your own python virtual environment.
- python version: 3.7.7 
- cuda version: 10.1.243

```
pip install -r requirements.txt
```

## Dataset Preparation
The tiny VOC dataset is come from the orignal Pascal VOC dataset and modified to smaller one.
get it from [here] (https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK) and rename the train_images/ folder to pascal_train/
The pascal_train.json file is the annotation of the training set.
test.json file is to generate the test result of this assignment.
Also, see [data_loader.ipynb](https://github.com/NCTU-VRDL/CS_IOC5008/blob/master/HW4/data_loader.ipynb) for more details.


### costom dataset
I reference this [dataset](https://github.com/Okery/PyTorch-Simple-MaskRCNN/blob/master/pytorch_mask_rcnn/datasets/coco_dataset.py) to processing my dataset and just read cocodatset.py to know more detail.
### data augmentaion
The original dataset only contains 1349 training images, and I directly double the image to 2698 and also add a set of taining images fliped vertically and horizontally-flip .Finally, the training images become 1349x3, two set of original images and one set of tranfromed images.
All the data augmentation processes are included in the cocodataset.py.



## Training

Rename the train_images/ folder to pascal_train/ before training.

Training the model, just run this code, and the model weights will be saved in the model/ folder.
```
python train.py
```
You are able to modify model.py to choose other backbone of Mask R-CNN or add dropout to the mask head and box head.

```
python train.py --backbone {backbone:resnet50, resnet101, resnet152} --dropout {dropout:bool} --GPU {GPU index}
```
You might chage the bath size, according to you GPU memory size.
The expected training times are:

 GPUs  | Backbone | data augmentation| Epoch | Training Time
------------ | ------------- | ------------- |--------------|--------
 1x TitanX  | res50 | original x1 + transform | 1 | 8.5 mins
 1x TitanX  | res101 | original x1 + transform | 1 | 10.5 mins
 1x TitanX  | res152 | original x1 + transform | 1 | 13.5 mins
 1x TitanX  | res152 | original x2 + transform | 1 | 21 mins
 
 
When starting running the code, you can see the ouput like this.
```
Epoch: [0]  [   0/1924]  eta: 0:32:50  lr: 0.000010  loss: 4.5240 (4.5240)  loss_classifier: 3.0904 (3.0904)  loss_box_reg: 0.0164 (0.0164)  loss_mask: 0.7192 (0.7192)  loss_objectness: 0.6921 (0.6921)  loss_rpn_box_reg: 0.0059 (0.0059)  time: 1.0241  data: 0.1882  max mem: 3784
Epoch: [0]  [ 200/1924]  eta: 0:18:18  lr: 0.001009  loss: 1.0625 (2.3294)  loss_classifier: 0.2010 (1.1289)  loss_box_reg: 0.0936 (0.0685)  loss_mask: 0.6801 (0.7244)  loss_objectness: 0.0616 (0.3558)  loss_rpn_box_reg: 0.0190 (0.0519)  time: 0.6329  data: 0.0069  max mem: 7144
Epoch: [0]  [ 400/1924]  eta: 0:16:27  lr: 0.002008  loss: 1.0838 (1.7313)  loss_classifier: 0.2225 (0.7003)  loss_box_reg: 0.1184 (0.0930)  loss_mask: 0.5629 (0.6806)  loss_objectness: 0.0356 (0.2127)  loss_rpn_box_reg: 0.0168 (0.0447)  time: 0.6757  data: 0.0071  max mem: 7292
Epoch: [0]  [ 600/1924]  eta: 0:14:19  lr: 0.003007  loss: 1.0832 (1.5320)  loss_classifier: 0.2381 (0.5494)  loss_box_reg: 0.0940 (0.0955)  loss_mask: 0.6604 (0.6767)  loss_objectness: 0.0723 (0.1683)  loss_rpn_box_reg: 0.0424 (0.0421)  time: 0.6344  data: 0.0070  max mem: 7298
Epoch: [0]  [ 800/1924]  eta: 0:12:11  lr: 0.004006  loss: 0.9845 (1.4262)  loss_classifier: 0.2008 (0.4835)  loss_box_reg: 0.1111 (0.1013)  loss_mask: 0.5405 (0.6558)  loss_objectness: 0.0337 (0.1438)  loss_rpn_box_reg: 0.0225 (0.0417)  time: 0.6553  data: 0.0069  max mem: 7981
Epoch: [0]  [1000/1924]  eta: 0:10:01  lr: 0.005000  loss: 0.9587 (1.3401)  loss_classifier: 0.2730 (0.4372)  loss_box_reg: 0.1100 (0.1023)  loss_mask: 0.5614 (0.6331)  loss_objectness: 0.0409 (0.1259)  loss_rpn_box_reg: 0.0389 (0.0415)  time: 0.6558  data: 0.0069  max mem: 7981
Epoch: [0]  [1200/1924]  eta: 0:07:51  lr: 0.005000  loss: 0.9868 (1.2725)  loss_classifier: 0.2256 (0.4075)  loss_box_reg: 0.0971 (0.1038)  loss_mask: 0.4540 (0.6070)  loss_objectness: 0.0377 (0.1133)  loss_rpn_box_reg: 0.0322 (0.0409)  time: 0.6339  data: 0.0070  max mem: 7981
Epoch: [0]  [1400/1924]  eta: 0:05:41  lr: 0.005000  loss: 1.1466 (1.2227)  loss_classifier: 0.3937 (0.3871)  loss_box_reg: 0.1528 (0.1049)  loss_mask: 0.4566 (0.5865)  loss_objectness: 0.0398 (0.1035)  loss_rpn_box_reg: 0.0296 (0.0408)  time: 0.6417  data: 0.0070  max mem: 7981
Epoch: [0]  [1600/1924]  eta: 0:03:31  lr: 0.005000  loss: 0.8330 (1.1808)  loss_classifier: 0.2412 (0.3725)  loss_box_reg: 0.0843 (0.1059)  loss_mask: 0.4456 (0.5674)  loss_objectness: 0.0352 (0.0950)  loss_rpn_box_reg: 0.0304 (0.0401)  time: 0.6532  data: 0.0069  max mem: 7981
Epoch: [0]  [1800/1924]  eta: 0:01:21  lr: 0.005000  loss: 0.8575 (1.1493)  loss_classifier: 0.2284 (0.3598)  loss_box_reg: 0.0919 (0.1061)  loss_mask: 0.4212 (0.5547)  loss_objectness: 0.0287 (0.0891)  loss_rpn_box_reg: 0.0156 (0.0396)  time: 0.6497  data: 0.0071  max mem: 7981
Epoch: [0]  [1923/1924]  eta: 0:00:00  lr: 0.005000  loss: 0.7987 (1.1317)  loss_classifier: 0.2179 (0.3515)  loss_box_reg: 0.0991 (0.1054)  loss_mask: 0.4594 (0.5483)  loss_objectness: 0.0298 (0.0865)  loss_rpn_box_reg: 0.0339 (0.0400)  time: 0.6180  data: 0.0070  max mem: 7981
Epoch: [0] Total time: 0:20:57 (0.6537 s / it)
creating index...
index created!
.
.
```
The following result is that I set the backbone network as resnet152. Dataset size is three times larger due to doubling the data and augmentation.
The validation output after one epoch:
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.042
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.004
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.010
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.018
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.019
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.049
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.054
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.080
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.039
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.004
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.005
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.021
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.022
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.056
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.059
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.010
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.027
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.085
```
After 1 epochs I obtain a COCO-style IoU=0.5 mAP of 3.9

After training for 10 epochs, I got the following metrics and the IoU=0.5 mAP is 33.5.
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.183
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.173
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.119
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.221
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.158
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.208
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.232
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.178
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.096
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.217
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.114
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.230
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.249
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.275
```



### Load trained parameters
I have save the trained model parameters [here](https://drive.google.com/file/d/1MPhXIAtHjOfyKTGkUygy5m7wSitJrJSd/view?usp=sharing), you could directly download it and save in ./model folder to do inference without retraining.


## Inference
Just run inference.py and you can inferece the test images provided from the [google drive](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK) and generate the json file of inference result with test.json file. 
```
python inference.py
```
use other model weight and backbone
```
python inference.py --backbone resnet152 -m mask_rcnn_res152_x3.pth -o submission.json
```
- -o: is the output json file name you want to set
The json format is a list of dictionaries.
Each dictionary contains four keys
- "image_id": id of test image, which is the key in “test.json”, int
- "score": probability for the class of this instance, float
- "category_id": category id of this instance, int
- “segmentation”: Encode the mask in RLE by provide function, str

You can also use the [demo.ipybn](https://github.com/sweiichen/tiny-pascal-voc/blob/main/demo.ipynb) to visualize the predict mask and get more detail of inference.





