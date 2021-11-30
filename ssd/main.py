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

from model import SSDLiteModel
from eval import predict


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


def main():
    ckpt_path = './weights/epoch=49-val_map50=0.39.ckpt'
    
    device = 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SSDLiteModel(num_classes=3)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model = model.load_from_checkpoint(checkpoint_path=ckpt_path)

    width = 1280
    height = 720
    
    cap = cv2.VideoCapture(0)
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
        bboxs, labels = predict(model, device, img, nm_thrs=0.5, score_thrs=0.8)
        
        if bboxs.size != 0:
            widths = bboxs[:, 2] - bboxs[:, 0]
            heights = bboxs[:, 3] - bboxs[:, 1]
            areas = widths * heights
            biggset_index = np.argmax(areas)
            big_x1, big_y1, big_x2, big_y2 = map(round, bboxs[biggset_index])
            big_w, big_h = big_x2 - big_x1, big_y2 - big_y1
            # string = 'X{0:d}Y{1:d}'.format((big_x1+big_w//2),(big_y1+big_h//2))
            string = f'Y{big_y1+big_h//2:d}'
            # print(string)
            ArduinoSerial.write(string.encode('utf-8'))
        
        for i, box in enumerate(bboxs):
            x, y, x2, y2= list(map(int, box))
            # print(box)
            # x, y, x2, y2 = box
            label = labels[i]
            w, h = x2-x, y2-y
            # print(x, y, w, h, label)

            # cv2.rectangle(raw_img, (x, y, w, h), 0, 1)
            # cv2.putText(raw_img, f'width: {w}, height: {h}', (900, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
                
            # cv2.putText(raw_img, f'{temperature}˚C', (900, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
            
            if label == 1:
                if w > 300 and h > 400:
                    temperature = ArduinoSerial.readlines()
                    if is_available(temperature)[0]:
                        temperature = is_available(temperature)[1]
                        print(temperature)
                        if temperature >= 37.5:
                            cv2.rectangle(raw_img, (x, y, w, h), (0, 0, 255), 1)
                            (tw, th), _ = cv2.getTextSize(f'mask: {temperature}C', cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
                            cv2.rectangle(raw_img, (x, y-20), (x+tw, y), (0, 0, 255), -1)
                            cv2.putText(raw_img, f'mask: {temperature}C', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.circle(raw_img, (x+w//2, y+h//2), 2, (0, 0, 255), 2)
                        elif temperature > 30.0 and temperature < 37.5:
                            cv2.rectangle(raw_img, (x, y, w, h), (0, 255, 0), 1)
                            (tw, th), _ = cv2.getTextSize(f'mask: {temperature}C', cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
                            cv2.rectangle(raw_img, (x, y-20), (x+tw, y), (0, 255, 0), -1)
                            cv2.putText(raw_img, f'mask: {temperature}C', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.circle(raw_img, (x+w//2, y+h//2), 2, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(raw_img, (x, y, w, h), (255, 255, 0), 1)
                            (tw, th), _ = cv2.getTextSize(f'mask: {temperature}C', cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
                            cv2.rectangle(raw_img, (x, y-20), (x+tw, y), (255, 255, 0), -1)
                            cv2.putText(raw_img, f'mask: {temperature}C', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.circle(raw_img, (x+w//2, y+h//2), 2, (255, 255, 0), 2)
                    else:
                        cv2.rectangle(raw_img, (x, y, w, h), (255, 0, 0), 1)
                        (tw, th), _ = cv2.getTextSize('mask', cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
                        cv2.rectangle(raw_img, (x, y-20), (x+tw, y), (255, 0, 0), -1)
                        cv2.putText(raw_img, 'mask', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(raw_img, (x+w//2, y+h//2), 2, (255, 0, 0), 2)
                else:
                    cv2.rectangle(raw_img, (x, y, w, h), (255, 0, 0), 1)
                    (tw, th), _ = cv2.getTextSize('mask', cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
                    cv2.rectangle(raw_img, (x, y-20), (x+tw, y), (255, 0, 0), -1)
                    cv2.putText(raw_img, 'mask', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(raw_img, (x+w//2, y+h//2), 2, (255, 0, 0), 2)
            elif label == 2:
                cv2.rectangle(raw_img, (x, y, w, h), 0, 1)
                (tw, th), _ = cv2.getTextSize('unweared_mask', cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
                cv2.rectangle(raw_img, (x, y-20), (x+tw, y), 0, -1)
                cv2.putText(raw_img, 'unweared_mask', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA) 
                cv2.circle(raw_img, (x+w//2, y+h//2), 2, 0, 2)   
                
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