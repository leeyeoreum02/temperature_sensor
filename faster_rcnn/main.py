import numpy as np # linear algebra
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches
import os
import cv2
import time
import serial


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


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def single_img_predict(model, device, img, nm_thrs = 0.5, score_thrs=0.9):
    model.eval()
    model.to(device)
    img = img / 255
    img = img.transpose((2, 0, 1))
    test_img = torch.tensor(img, dtype=torch.float32)
    test_img = test_img.to(device)
    with torch.no_grad():
        predictions = model([test_img])
        
    # non-max supression
    keep_boxes = torchvision.ops.nms(predictions[0]['boxes'].cpu(),predictions[0]['scores'].cpu(), nm_thrs)
    
    # Only display the bounding boxes which higher than the threshold
    score_filter = predictions[0]['scores'].cpu().numpy()[keep_boxes] > score_thrs
    
    # get the filtered result
    test_boxes = predictions[0]['boxes'].cpu().numpy()[keep_boxes][score_filter]
    test_labels = predictions[0]['labels'].cpu().numpy()[keep_boxes][score_filter]
    
    return test_boxes, test_labels


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
        1: 'mask',
        2: 'unweared_mask',
    }
    
    cv2.rectangle(raw_img, (x, y, w, h), color, 1)
    (tw, th), _ = cv2.getTextSize(categories[label], cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
    cv2.rectangle(raw_img, (x, y-20), (x+tw, y), color, -1)
    cv2.putText(raw_img, categories[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(raw_img, (x+w//2, y+h//2), 2, color, 2)
    
    
def draw_temperature(message, raw_img, frame_height, frame_width):
    (tw, th), _ = cv2.getTextSize(message, 
        cv2.FONT_HERSHEY_COMPLEX, 3, 2)
    output = cv2.rectangle(
        raw_img, 
        (0, frame_height-100), 
        (frame_width, frame_height-100+th), 
        (0, 255, 0), -1)
    cv2.addWeighted(raw_img, 1, output, 0.5, 0)
    cv2.putText(raw_img, message, 
        (frame_width//4, frame_height-100+th//4), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        3, (255, 255, 255), 2, cv2.LINE_AA)
    
    
def get_temperature(arduino_serial, frame_width, frame_height, raw_img, bbox):
    _, _, w, h = bbox
        
    if w > 300 and h > 400:
        temperature = arduino_serial.readlines()
        
        if not is_available(temperature)[0]:
            draw_temperature('Measuring...', 
                raw_img, frame_height, frame_width)
            return
            
        temperature = is_available(temperature)[1]
            
        if temperature < 30.0:
            draw_temperature('Please come close.', 
                raw_img, frame_height, frame_width)
        else:
            draw_temperature(str(temperature), 
                raw_img, frame_height, frame_width)


def main():
    model = get_model_instance_segmentation(3)
    model.load_state_dict(torch.load('./weights/model_mobileV3_27_0.220.pth'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    width = 1280
    height = 720
    # cap = cv2.VideoCapture(0) # desktop
    cap = cv2.VideoCapture(1) # notebook
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ArduinoSerial = serial.Serial('com3', 9600, timeout=5e-3)
    # ArduinoSerial = serial.Serial('com3', 11520, timeout=5e-1)

    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        ret, raw_img = cap.read()
        raw_img = cv2.flip(raw_img, 1)
        # fps
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        fps = round(fps)
        cv2.putText(raw_img, f'FPS: {fps}', (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
        prev_frame_time = new_frame_time
        
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        bboxs, labels = single_img_predict(model, device, img)
        
        if bboxs.size != 0:
            widths = bboxs[:, 2] - bboxs[:, 0]
            heights = bboxs[:, 3] - bboxs[:, 1]
            areas = widths * heights
            biggset_index = np.argmax(areas)
            big_x1, big_y1, big_x2, big_y2 = map(round, bboxs[biggset_index])
            big_w, big_h = big_x2 - big_x1, big_y2 - big_y1
            string = f'Y{big_y1+big_h//2:d}'
            ArduinoSerial.write(string.encode('utf-8'))
        
        for i, box in enumerate(bboxs):
            x, y, x2, y2= list(map(int, box))
        
            label = labels[i]
            w, h = x2-x, y2-y
            
            if label == 1:
                draw_bbox(raw_img, (x, y, w, h), 'green', label)
            else:
                draw_bbox(raw_img, (x, y, w, h), 'red', label)
                
            get_temperature(ArduinoSerial, width, height, raw_img, (x, y, w, h))
                
        cv2.rectangle(raw_img, (width//2-100, height//2-100),
                    (width//2+100, height//2+100),
                    (255,255,255), 3)
        cv2.imshow('mask_detect', raw_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    cap.release()
    cv2.destroyAllWindows()
    ArduinoSerial.close()


if __name__ == '__main__':
    main()