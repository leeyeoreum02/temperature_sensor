import argparse

from model import SSDLiteModel
from dataset import SSDLiteDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


parser = argparse.ArgumentParser(description='Training SSDLite')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()


def get_train_transforms():
    return A.Compose(
        [
            # A.Resize(height=720, width=1024),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomContrast(p=0.2),
            A.Blur(p=0.2),
            A.MedianBlur(p=0.2),
            A.ColorJitter(p=0.2),
            A.GaussianBlur(p=0.2),
            A.HorizontalFlip(p=0.2),
            A.RandomCrop(height=150, width=150, p=0.4),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['labels'],
        ),
    )


def get_valid_transforms():
    return A.Compose(
        [
            # A.Resize(height=720, width=1024),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['labels']
        ),
    )


def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    data_module = SSDLiteDataModule(
        root_path='../data/',
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        train_transforms=get_train_transforms(),
        valid_transforms=get_valid_transforms(),
    )

    model = SSDLiteModel(
        num_classes=3,
        learning_rate=args.lr,
        pretrained=True,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_map50',
        dirpath='./weights/',
        filename='{epoch}-{val_map50:.2f}-aug++',
        save_top_k=-1,
        mode='max',
        save_weights_only=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        # gpus=[0, 1, 2, 3],
        gpus=1,
        precision=16,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
