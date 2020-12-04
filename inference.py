import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import json
from cocodataset import COCODataset
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from model import get_instance_segmentation_model
from itertools import groupby
from pycocotools import mask as maskutil
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--GPU",type=int, default=1)
parser.add_argument("--backbone",type=str,default="resnet152")
parser.add_argument("-o",dest="output", type=str,default="submission.json")
parser.add_argument("-m",dest="model", type=str,default="mask_rcnn_res152_x3.pth")
args = parser.parse_args()


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

def mask_to_binary(mask, threshold=0.1):
    t = Variable(torch.Tensor([threshold]))
    binary_mask = (mask > t).int()
    return binary_mask

device = torch.device(f'cuda:{args.GPU}') if torch.cuda.is_available() else torch.device('cpu')
mask_threshold = 0.3 # every pixel of predict mask is probability vale, set a threshold to convert the value to binary
score_threshold = 0.4 # the confidence score of an object
checkpoint = torch.load(f'model/{args.model}',map_location=device) # load the model trained weights
num_classes = 21

model = get_instance_segmentation_model(num_classes, args.backbone)

model = model.to(device)
model.load_state_dict(checkpoint["model_state_dict"])
cocoGt = COCO("test.json")
model.eval()
coco_dt = []

for imgid in cocoGt.imgs:
    image = Image.open("./test_images/" + cocoGt.imgs[imgid]['file_name'])
    image = torchvision.transforms.ToTensor()(image)
    # run inference
    with torch.no_grad():
        prediction = model([image.to(device)])
    n_instances = len(prediction[0]['scores'])  
    if len(prediction[0]['labels']) > 0: 
        for i in range(n_instances): # Loop all instances
                # save information of the instance in a dictionary then append on coco_dt list
            if prediction[0]['scores'][i] > score_threshold:
                pred = {}
                pred['image_id'] = imgid # this imgid must be same as the key of test.json
                pred['category_id'] = int(prediction[0]['labels'][i])
                mask = mask_to_binary(prediction[0]['masks'][i, 0].cpu(),threshold=mask_threshold)
                pred['segmentation'] = binary_mask_to_rle(mask.numpy()) # save binary mask to RLE, e.g. 512x512 -> rle
                pred['score'] = float(prediction[0]['scores'][i])
                coco_dt.append(pred)

    
print(coco_dt[-1])
with open(f"{args.output}", "w") as f:
    json.dump(coco_dt, f)