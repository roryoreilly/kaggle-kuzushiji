import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList

class KuzushijiDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.name_to_id = None
        self.id_to_name = None
        self._imgpath = os.path.join('/kaggle/input/kuzushiji-recognition/train_images', '%s.jpg')
        self.annotations = pd.read_csv(os.path.join('/kaggle/input/kuzushiji-recognition', 'train.csv'))
        self.ids = self.annotations['image_id'].values.tolist()

        self.CLASSES = pd.read_csv(os.path.join('/kaggle/input/kuzushiji-recognition', 'unicode_translation.csv'))[
            'Unicode'].values.tolist()
        self.CLASSES.insert(0, "__background__")
        cls = self.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(self._imgpath % img_id)
        width, height = img.size
        anno = self._preprocess_annotation(self.annotations.loc[idx])

        target = BoxList(anno["boxes"], (width, height), mode="xywh")
        target.add_field("labels", anno["labels"])

        if self.transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def _preprocess_annotation(self, anno):
        labels = np.array(anno['labels'].split(' ')).reshape(-1, 5)
        boxes = []
        gt_classes = []
        for label in labels:
            boxes.append(list(map(int, label[1:])))
            gt_classes.append(self.class_to_ind[label[0]])
        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes)
        }
        return res

    def get_img_info(self, idx):
        img_id = self.ids[idx]
        img = Image.open(self._imgpath % img_id)
        width, height = img.size
        return {"height": height, "width": width}

    def __len__(self):
        return len(self.ids)