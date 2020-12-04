import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torchvision import transforms
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform():
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    # transforms.append(T.ToTensor())
    # if train:
    #     # during training, randomly flip the training images
    #     # and ground-truth for data augmentation
    transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class GeneralizedDataset:
    """
    Main class for Generalized Dataset.
    """
    
    def __init__(self, max_workers=2, verbose=False, transforms=False):
        self.max_workers = max_workers
        self.verbose = verbose
        self.transforms = get_transform()
            
    def __getitem__(self, i):
        img_id = self.ids[i]
        image = self.get_image(img_id)
        image = transforms.ToTensor()(image)
        target = self.get_target(img_id) if self.train else {}
        if "_aug" in img_id:
          image, target = self.transforms(image, target)
        return image, target   
    
    def __len__(self):
        return len(self.ids)
    
    def check_dataset(self, checked_id_file):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in method `_check`.
        """
        
        if os.path.exists(checked_id_file):
            info = [line.strip().split(", ") for line in open(checked_id_file)]
            self.ids, self.aspect_ratios = zip(*info)
            return
        
        since = time.time()
        print("Checking the dataset...")
        
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        seqs = torch.arange(len(self)).chunk(self.max_workers)
        tasks = [executor.submit(self._check, seq.tolist()) for seq in seqs]

        outs = []
        for future in as_completed(tasks):
            outs.extend(future.result())
        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        outs.sort(key=lambda x: self.id_compare_fn(x[0]))
        
        with open(checked_id_file, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))
         
        info = [line.strip().split(", ") for line in open(checked_id_file)]
        self.ids, self.aspect_ratios = zip(*info)
        print("checked id file: {}".format(checked_id_file))
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))
        
    def _check(self, seq):
        out = []
        for i in seq:
            img_id = self.ids[i]
            target = self.get_target(img_id)
            boxes = target["boxes"]
            labels = target["labels"]
            masks = target["masks"]

            try:
                assert len(boxes) > 0, "{}: len(boxes) = 0".format(i)
                assert len(boxes) == len(labels), "{}: len(boxes) != len(labels)".format(i)
                assert len(boxes) == len(masks), "{}: len(boxes) != len(masks)".format(i)

                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                if self.verbose:
                    print(img_id, e)
        return out

from PIL import Image

import torch
# from .generalized_dataset import GeneralizedDataset
       
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        ann_file = os.path.join(data_dir, "{}.json".format(split))
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]+[str(k)+"_aug" for k in self.coco.imgs]+[str(k)+"_2" for k in self.coco.imgs]
        # self.ids = [str(k) for k in self.coco.imgs]
        # print(len(self.ids))
        
        self._classes = {k: v["name"] for k, v in self.coco.cats.items()}
        self.classes = tuple(self.coco.cats[k]["name"] for k in sorted(self.coco.cats))
        # resutls' labels convert to annotation labels
        # self.ann_labels = {self.classes.index(v): k for k, v in self._classes.items()}
        
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            # self.check_dataset(checked_id_file)
        
    def get_image(self, img_id):
        if "_aug" in img_id:
            img_id = img_id[:-4]
        if "_2" in img_id:
            img_id = img_id[:-2]
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"]))
        return image.convert("RGB")
    
    @staticmethod
    def convert_to_xyxy(box): # box format: (xmin, ymin, w, h)
        new_box = torch.zeros_like(box)
        new_box[:, 0] = box[:, 0]
        new_box[:, 1] = box[:, 1]
        new_box[:, 2] = box[:, 0] + box[:, 2]
        new_box[:, 3] = box[:, 1] + box[:, 3]
        return new_box # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id):
        if "_aug" in img_id:
          img_id = img_id[:-4]
        if "_2" in img_id:
            img_id = img_id[:-2]
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []
        areas = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                name = self._classes[ann["category_id"]]
                
                labels.append(int(ann["category_id"]))
                
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)
                areas.append(ann['area'])
                
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
            iscrowd = torch.zeros((len(anns),), dtype=torch.int64)
            areas = torch.tensor(areas)
            

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks, iscrowd=iscrowd, area=areas)
        return target