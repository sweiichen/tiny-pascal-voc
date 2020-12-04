import torch
import utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from engine import train_one_epoch, evaluate
from cocodataset import COCODataset
from datetime import datetime
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, p=0.5, training=self.training)

        return x

class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            d["dropout{}".format(layer_idx)] = nn.Dropout(0.5)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
         

      
def get_instance_segmentation_model(num_classes, backbone, dropout=False):
    # load an instance segmentation model where backbone is pretrained ImageNet
    backbone = resnet_fpn_backbone(backbone, pretrained=True)
    model = MaskRCNN(backbone, num_classes)
    
    
    if dropout:
        # add drop out after FC layer of box head
        resolution = model.roi_heads.box_roi_pool.output_size[0]
        representation_size = 1024
        model.roi_heads.box_head = TwoMLPHead(backbone.out_channels * resolution ** 2,representation_size)
        # add drop out in mask head
        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        model.roi_heads.mask_head = MaskRCNNHeads(backbone.out_channels, mask_layers, mask_dilation)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model