from functools import partial
from typing import Optional, Callable, Any

from torch import nn, optim
# from torchvision.ops import box_iou
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import _mobilenet_extractor, SSDLiteHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
import torchvision.models.detection._utils as det_utils
from torchmetrics.detection.map import Metric, MAP
from pytorch_lightning import LightningModule


# def _evaluate_iou(target, pred):
#     """
#     Evaluate intersection over union (IOU) for target from dataset and output prediction
#     from model
#     """
#     if pred["boxes"].shape[0] == 0:
#         # no box detected, 0 IOU
#         return torch.tensor(0.0, device=pred["boxes"].device)
#     return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class SSDLiteModel(LightningModule):
    def __init__(
        self,
        learning_rate=0.0002,
        num_classes=2,
        pretrained: bool = True,
        pretrained_backbone: bool = False,
        metric: Metric = MAP,
        **kwargs: Any,
    ):
        super().__init__()
        self.lr = learning_rate
        self.model = self.get_model(pretrained=pretrained, num_classes=num_classes)
        self.metric = metric()
        
    def get_model(self, pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                  pretrained_backbone: bool = False, trainable_backbone_layers: Optional[int] = None,
                  norm_layer: Optional[Callable[..., nn.Module]] = None,
                  **kwargs: Any):
        model = ssdlite320_mobilenet_v3_large(pretrained=pretrained)
        
        if pretrained:
            trainable_backbone_layers = _validate_trainable_layers(
            pretrained or pretrained_backbone, trainable_backbone_layers, 6, 6)
            
            pretrained_backbone = False
            
            reduce_tail = not pretrained_backbone
            
            if norm_layer is None:
                norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

            backbone = _mobilenet_extractor("mobilenet_v3_large", progress, pretrained_backbone, trainable_backbone_layers,
                                            norm_layer, reduced_tail=reduce_tail, **kwargs)
            
            size = (320, 320)
            anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
            out_channels = det_utils.retrieve_out_channels(backbone, size)
            num_anchors = anchor_generator.num_anchors_per_location()
            
            model.head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)
        
        return model
        
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
        # return torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.lr,
        #     momentum=0.9,
        #     weight_decay=0.005,
        # )

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # print(images, targets)

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        self.log(
            'box_loss', loss_dict['bbox_regression'].detach(), 
            prog_bar=True, logger=True
        )
        self.log(
            'cls_loss', loss_dict['classification'].detach(), 
            prog_bar=True, logger=True
        )
        
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        # iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        self.metric.update(outs, targets)
        
        avg_map = self.metric.compute()
        self.log('val_map50', avg_map['map_50'].detach(), 
            prog_bar=True, logger=True
        )
        
    def validation_epoch_end(self, outs):
        # avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        avg_map = self.metric.compute()
        
        logs = {"val_map50": avg_map['map_50'].detach()}
        
        self.metric.reset()
        
        return {"avg_val_iou": avg_map, "log": logs}