import os
# import cv2
import json
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


torch.multiprocessing.set_sharing_strategy('file_system')


def _collate_fn(batch):
    return tuple(zip(*batch))


class FaceTrainDataset(Dataset):
    def __init__(self, image_dir, annot_path, transforms=None, **kwargs):
        with open(annot_path, 'r') as f:
            annotations = json.load(f)

        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = {annot['id']: {
            'file_name': annot['file_name'], 
            'height': annot['height'],
            'width': annot['width']
        } for annot in annotations['images']}
        # print(self.image_ids)

        self.annotations = defaultdict(list)
        for annot in annotations['annotations']:
            self.annotations[annot['image_id']].append({
                'category_id': annot['category_id'],
                'bbox': np.array(annot['bbox']),
                'area': annot['area']
            })

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.image_dir, image_id['file_name'])).convert('RGB')
        # image = cv2.imread(os.path.join(self.image_dir, image_id['file_name']), cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width, height = image_id['width'], image_id['height']

        boxes = []
        labels = []
        # areas = []
        for annot in self.annotations[index]:
            x1, y1, w, h = annot['bbox']
            x2, y2 = x1 + w, y1 + h
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, width)
            y2 = min(y2, height)
                
            boxes.append([x1, y1, x2, y2])
            labels.append(annot['category_id'])

        boxes = np.asarray(boxes)
        labels = np.asarray(labels)

        samples = {
            'image': np.asarray(image, dtype=np.uint8),
            # 'image': image.astype(np.uint8),
            'bboxes': boxes,
            'labels': labels
        }

        if self.transforms is not None:
            transformed = self.transforms(**samples)

        image = transformed['image']
        del transformed['image']
        
        target = {
            'boxes': torch.as_tensor(samples['bboxes'], dtype=torch.float32),
            'labels': torch.as_tensor(samples['labels'], dtype=torch.int64),
        }
        del samples

        return image, target
    
    
class SSDLiteDataModule(LightningDataModule):
    def __init__(
        self,
        root_path='../data/',
        train_transforms=None,
        valid_transforms=None,
        num_workers=4,
        batch_size=8
    ):
        super().__init__()
        self.root_path = root_path
        self.train_image_path = os.path.join(root_path, 'train', 'images')
        self.train_annot_path = os.path.join(root_path, 'train', 'annotations.json')
        self.valid_image_path = os.path.join(root_path, 'valid', 'images')
        self.valid_annot_path = os.path.join(root_path, 'valid', 'annotations.json')
        
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = FaceTrainDataset(self.train_image_path, self.train_annot_path, self.train_transforms)
        self.valid_dataset = FaceTrainDataset(self.valid_image_path, self.valid_annot_path, self.valid_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )
    

if __name__ == '__main__':
    # dataset = EfficientDetDataset('./data/', './data/annotations.json', get_train_transforms())
    # print(dataset[1])
    data_loader = SSDLiteDataModule()
    print(data_loader.on_train_dataloader)