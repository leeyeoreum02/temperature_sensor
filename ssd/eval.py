from albumentations.augmentations import transforms
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
import albumentations as A

from model import SSDLiteModel


def test_transforms(img):
    transformed = A.Compose(
        [
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    )(image=img)
    
    return transformed['image']


def predict(model, device, img, nm_thrs = 0.5, score_thrs=0.9):
    test_img = test_transforms(img)
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