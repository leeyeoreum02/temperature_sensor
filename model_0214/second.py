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
    # test_img = transforms.ToTensor()(img)
    model.eval()
    model.to(device)
    img = img / 255
    img = img.transpose((2, 0, 1))
    # print(img.shape)
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


def main():
    model = get_model_instance_segmentation(3)
    model.load_state_dict(torch.load('./weights/model_mobileV3_27_0.220.pth'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        bboxs, labels = single_img_predict(model, device, img)
        # print(ArduinoSerial.readlines())
        
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
            
            if label == 2:
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
            elif label == 1:
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