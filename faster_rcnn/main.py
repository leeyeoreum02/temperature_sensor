import numpy as np # linear algebra
import cv2
import serial
import sys
sys.path.append('../utils')

import torch

from eval import single_img_predict, get_model
from utils import draw_below, draw_above
from utils import draw_bbox, get_temperature


def main():
    model = get_model(3)
    model.load_state_dict(torch.load('./weights/model_mobileV3_27_0.220.pth'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    width = 1280
    height = 720
    
    cap = cv2.VideoCapture(0) # desktop
    # cap = cv2.VideoCapture(1) # notebook
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ArduinoSerial = serial.Serial('com3', 9600, timeout=5e-3)
    # ArduinoSerial = serial.Serial('com3', 11520, timeout=5e-1)

    while cap.isOpened():
        ret, raw_img = cap.read()
        raw_img = cv2.flip(raw_img, 1)
        
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
            
            if labels[biggset_index] == 1:
                draw_above('Please wear a mask!!!', raw_img, 'red')
        
            for i, box in enumerate(bboxs):
                x, y, x2, y2= list(map(int, box))
            
                label = labels[i]
                w, h = x2-x, y2-y
                
                if label == 2:
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