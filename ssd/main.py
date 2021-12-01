import numpy as np # linear algebra
import cv2
import time
import serial

import torch

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
    (tw, th), _ = cv2.getTextSize(categories[label], cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
    cv2.rectangle(raw_img, (x, y-20), (x+tw, y), color, -1)
    cv2.putText(raw_img, categories[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
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
    ckpt_path = './weights/epoch=99-val_map50=0.48.ckpt'
    
    device = 'cuda'
    model = SSDLiteModel(num_classes=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])

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
        bboxs, labels = predict(model, device, img, nm_thrs=0.2, score_thrs=0.9)
        
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