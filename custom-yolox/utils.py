import os
import json
import shutil
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch


def all_gt_pascal2coco(
    root_path: os.PathLike,
    save_path: os.PathLike = 'data',
) -> None:
    ret = defaultdict(list)
    xml_paths = glob(os.path.join(root_path, 'annotations', '*.xml'))
    
    categories = {
        'with_mask': 1,
        'without_mask': 2,
        'mask_weared_incorrect': 2,
    }
    
    n_id = 0
    for annot_path in tqdm(xml_paths):
        tree = ET.parse(annot_path)
        root = tree.getroot()
        
        height = int(root.find('size').findtext('height'))
        width = int(root.find('size').findtext('width'))
        file_name = root.findtext('filename')
        img_id = int(file_name.split('.')[0][-1])
        
        ret['images'].append({
            'file_name': file_name,
            'height': height,
            'width': width,
            'id': img_id,
        })
        
        for obj in root.findall('object'):
            label = obj.findtext('name')
            class_id = categories[label]
            # print(class_id)
            tags = (i.tag for i in obj.find('bndbox'))
            x1, y1, x2, y2 = map(int, (obj.find('bndbox').findtext(tag) for tag in tags))
            w, h = x2 - x1, y2 - y1
            
            ret['annotations'].append({
                'id': n_id,
                'image_id': img_id,
                'category_id': class_id,
                'area': w * h,
                'bbox': [x1, y1, w, h],
                'iscrowd': 0,
            })
            n_id += 1
    
    for name, id in categories.items():
        ret['categories'].append({
            'id': id,
            'name': name,
        })
    
    with open(save_path, 'w') as f:
        json.dump(ret, f)
        
        
def split_train_valid(
    root_path: os.PathLike, 
    train_path: os.PathLike, 
    valid_path: os.PathLike,
    split_rate: float = 0.2,
    shuffle: bool = True
) -> None:
    xml_path = os.path.join(root_path, 'annotations')
    img_path = os.path.join(root_path, 'images')
    
    xml_files = [file_name for file_name in os.listdir(xml_path)
                            if file_name.endswith(r'.xml')]
    
    if shuffle:
        random.Random(42).shuffle(xml_files)
        
    img_files = [f'{file_name.split(".")[0]}.png' for file_name in xml_files]
        
    cut_point = int(split_rate * len(xml_files))
    valid_xml_files = xml_files[:cut_point]
    train_xml_files = xml_files[cut_point:]
    valid_img_files = img_files[:cut_point]
    train_img_files = img_files[cut_point:]
    
    xml_train_path = os.path.join(train_path, 'annotations')
    if not os.path.exists(xml_train_path):
        os.makedirs(xml_train_path)
    else:
        shutil.rmtree(xml_train_path)
        os.makedirs(xml_train_path) 
    for xml_file in tqdm(train_xml_files):
        xml_src = os.path.join(xml_path, xml_file)
        xml_dst = os.path.join(xml_train_path, xml_file)
        shutil.copy(xml_src, xml_dst)
        
    img_train_path = os.path.join(train_path, 'images')
    if not os.path.exists(img_train_path):
        os.makedirs(img_train_path)
    else:
        shutil.rmtree(img_train_path)
        os.makedirs(img_train_path)
    for img_file in tqdm(train_img_files):
        img_src = os.path.join(img_path, img_file)
        img_dst = os.path.join(img_train_path, img_file)
        shutil.copy(img_src, img_dst)

    xml_valid_path = os.path.join(valid_path, 'annotations')
    if not os.path.exists(xml_valid_path):
        os.makedirs(xml_valid_path)
    else:
        shutil.rmtree(xml_valid_path)
        os.makedirs(xml_valid_path) 
    for xml_file in tqdm(valid_xml_files):
        xml_src = os.path.join(xml_path, xml_file)
        xml_dst = os.path.join(xml_valid_path, xml_file)
        shutil.copy(xml_src, xml_dst)
        
    img_valid_path = os.path.join(valid_path, 'images')
    if not os.path.exists(img_valid_path):
        os.makedirs(img_valid_path)
    else:
        shutil.rmtree(img_valid_path)
        os.makedirs(img_valid_path)
    for img_file in tqdm(valid_img_files):
        img_src = os.path.join(img_path, img_file)
        img_dst = os.path.join(img_valid_path, img_file)
        shutil.copy(img_src, img_dst)
        
        
def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

        
if __name__ == '__main__':
    root_path = '../data/'
    train_path = '../data/train/'
    valid_path = '../data/valid/'
    # split_train_valid(root_path, train_path, valid_path)
    all_gt_pascal2coco(root_path + 'train', root_path + 'train/annotations.json')
    all_gt_pascal2coco(root_path + 'valid', root_path + 'valid/annotations.json')