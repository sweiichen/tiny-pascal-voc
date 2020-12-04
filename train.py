import torch
import utils
import torchvision
from engine import train_one_epoch, evaluate
from cocodataset import COCODataset
from datetime import datetime
from collections import OrderedDict
from model import get_instance_segmentation_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--GPU", type=int, default=0)
parser.add_argument("--backbone", type=str, default="resnet152")
parser.add_argument("--dropout", type=bool, default=False)

args = parser.parse_args()
dataset = COCODataset("./", "pascal_train", train=True)
dataset_test = COCODataset("./", "pascal_train", train=True)
print(len(dataset.ids))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-200])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-200:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=2,
                                          collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=2,
                                               shuffle=False,
                                               num_workers=2,
                                               collate_fn=utils.collate_fn)

device = torch.device(
    f'cuda:{args.GPU}') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 21

# get the model using our helper function
model = get_instance_segmentation_model(num_classes, args.backbone,
                                        args.dropout)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,
                            lr=0.005,
                            momentum=0.9,
                            weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations

    train_one_epoch(model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    print_freq=200)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        },
        f"model/mask_rcnn_{datetime.now().month}{datetime.now().day}_{arg.backbone}.pth"
    )
