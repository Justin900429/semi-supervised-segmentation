import glob
import os
from typing import Tuple

import albumentations as albu
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .seg_dataset import SegDataset


def create_data_loader(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    num_classes: int = 10,
    img_size: int = 512,
    use_ssl: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    data_list = glob.glob(os.path.join(root, "images", "*.jpg"))

    labeled_data_list = []
    unlabeled_data_list = []
    for data in data_list:
        label_mask_name = data.replace("images", "segment").replace(".jpg", "_mask.png")
        if os.path.exists(label_mask_name):
            labeled_data_list.append(data)
        else:
            unlabeled_data_list.append(data)

    train_list, val_list = train_test_split(
        labeled_data_list, test_size=20, random_state=42
    )
    weak_transform, strong_transform = get_semi_supervised_transform(img_size)
    train_dataset = SegDataset(
        labeled_data_list=train_list,
        unlabeled_data_list=unlabeled_data_list,
        num_classes=num_classes,
        train=True,
        labeled_transform=get_train_transform(img_size),
        unlabeled_weak_transform=weak_transform,
        unlabeled_strong_transform=strong_transform,
        use_ssl=use_ssl,
    )
    val_dataset = SegDataset(
        labeled_data_list=val_list,
        num_classes=num_classes,
        train=False,
        labeled_transform=get_validation_transform(img_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_test_transform(img_size: int) -> T.Compose:
    test_transform = [
        T.ToTensor(),
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize([0.5], [0.5]),
    ]
    return T.Compose(test_transform)


def get_train_transform(img_size: int) -> albu.Compose:
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=0.3,
            border_mode=0,
        ),
        albu.Perspective(p=0.5),
        albu.Resize(
            height=img_size,
            width=img_size,
            always_apply=True,
        ),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.6,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.6,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1, always_apply=True),
                albu.HueSaturationValue(p=1, always_apply=True),
            ],
            p=0.6,
        ),
    ]
    return albu.Compose(train_transform)


def get_semi_supervised_transform(img_size: int) -> albu.Compose:
    common_train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=0.3,
            border_mode=0,
            interpolation=cv2.INTER_NEAREST,
        ),
        albu.Perspective(p=0.5, interpolation=cv2.INTER_NEAREST),
        albu.Resize(
            height=img_size,
            width=img_size,
            always_apply=True,
            interpolation=cv2.INTER_NEAREST,
        ),
    ]

    strong_train_tramsform = [
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.6,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.6,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.6,
        ),
    ]
    return albu.Compose(
        common_train_transform, additional_targets={"strong": "image"}
    ), albu.Compose(strong_train_tramsform)


def get_validation_transform(img_size: int) -> albu.Compose:
    test_transform = [albu.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST)]
    return albu.Compose(test_transform)
