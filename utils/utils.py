import os
import cv2
import json
import random
import shutil
from tqdm import tqdm
from glob import glob
import xml.etree.ElementTree as ET
from collections import defaultdict


def is_available(temperature):
    if temperature:
        temperature = str(temperature[0]).split("'")[1][:4]
        try:
            temperature = float(temperature)
            return [True, temperature]
        except ValueError:
            return [False]
    else:
        return [False]


def get_color(color):
    colors = {
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'sky': (255, 255, 0),
        'black': (0, 0, 0),
    }
    return colors[color]
    
    
def draw_bbox(raw_img, bbox, color, label):
    x, y, w, h = bbox
    color = get_color(color)
    
    categories = {
        2: 'mask',
        1: 'unweared_mask',
    }
    
    cv2.rectangle(raw_img, (x, y, w, h), color, 1)
    (tw, th), _ = cv2.getTextSize(categories[label], cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
    cv2.rectangle(raw_img, (x, y-20), (x+tw, y), color, -1)
    cv2.putText(raw_img, categories[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(raw_img, (x+w//2, y+h//2), 2, color, 2)
    
    
def draw_below(message, raw_img, frame_height, frame_width, color):
    color = get_color(color)
    (tw, th), _ = cv2.getTextSize(message, 
        cv2.FONT_HERSHEY_COMPLEX, 3, 2)
    cv2.rectangle(
        raw_img, 
        (0, frame_height-100), 
        (frame_width, frame_height-100+th), 
        color, -1)
    cv2.putText(raw_img, message, 
        (frame_width//4, frame_height-100+th//4), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        3, (255, 255, 255), 2, cv2.LINE_AA)
    

def draw_above(message, raw_img, color):
    color = get_color(color)
    (tw, th), _ = cv2.getTextSize(message, 
        cv2.FONT_HERSHEY_COMPLEX, 3, 2)
    cv2.putText(raw_img, message, 
        (100, th+5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        3, color, 2, cv2.LINE_AA)
    
    
def get_temperature(arduino_serial, frame_width, frame_height, raw_img, bbox):
    _, _, w, h = bbox
        
    if w > 300 and h > 400:
        temperature = arduino_serial.readlines()
        
        if not is_available(temperature)[0]:
            draw_below('Measuring...', 
                raw_img, frame_height, frame_width, 'green')
            return
            
        temperature = is_available(temperature)[1]
            
        if temperature < 30.0:
            draw_below('Please come close.', 
                raw_img, frame_height, frame_width, 'green')
        else:
            draw_below(str(temperature), 
                raw_img, frame_height, frame_width, 'green')

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
    for img_id, annot_path in tqdm(enumerate(xml_paths)):
        tree = ET.parse(annot_path)
        root = tree.getroot()
        
        height = int(root.find('size').findtext('height'))
        width = int(root.find('size').findtext('width'))
        file_name = root.findtext('filename')
        # img_id = int(file_name[12:].split('.')[0])
        
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
    
    ret['categories'].append({
        'id': 0,
        'name': '__background__',
    })
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