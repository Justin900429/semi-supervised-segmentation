import os
import random
from typing import Callable, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

COLOR_LIST = [
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]


class SegDataset(Dataset):
    def __init__(
        self,
        labeled_data_list: List[str],
        num_classes: int,
        unlabeled_data_list: List[str] = None,
        train: bool = True,
        labeled_transform: Callable = None,
        unlabeled_weak_transform: Callable = None,
        unlabeled_strong_transform: Callable = None,
        use_ssl: bool = True,
    ):
        self.labeled_data_list = labeled_data_list
        self.unlabeled_data_list = unlabeled_data_list

        self.train = train
        self.unlabeled_weak_transform = unlabeled_weak_transform
        self.unlabeled_strong_transform = unlabeled_strong_transform
        self.labeled_transform = labeled_transform
        self.num_classes = num_classes
        self.normalization = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.use_ssl = use_ssl

    def __len__(self):
        return len(self.labeled_data_list)

    def __getitem__(self, idx):
        # ======= Load label pair ========
        img = cv2.imread(self.labeled_data_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_mask_name = (
            self.labeled_data_list[idx]
            .replace("images", "segment")
            .replace(".jpg", "_mask.png")
        )
        label_mask = cv2.imread(label_mask_name, 0)

        if self.labeled_transform is not None:
            transformed = self.labeled_transform(image=img, mask=label_mask)
            img = transformed["image"]
            label_mask = transformed["mask"]
        img = self.normalization(img)
        label_mask = torch.from_numpy(label_mask).long()
        label_mask = torch.nn.functional.one_hot(
            label_mask, num_classes=self.num_classes
        ).permute(2, 0, 1)

        if not self.train:
            return img, label_mask

        # ======= Load unlabel image ========
        if not self.use_ssl:  # dummy data, use `ShortTensor` to save memory
            weak_unlabeled_img = torch.ShortTensor([0])
            strong_unlabeled_img = torch.ShortTensor([0])
        else:
            unlabeled_img = random.choice(self.unlabeled_data_list)
            unlabeled_img = cv2.imread(unlabeled_img)
            unlabeled_img = cv2.cvtColor(unlabeled_img, cv2.COLOR_BGR2RGB)

            weak_unlabeled_img = np.copy(unlabeled_img)
            strong_unlabeled_img = np.copy(unlabeled_img)
            if self.unlabeled_weak_transform is not None:
                transformed = self.unlabeled_weak_transform(
                    image=weak_unlabeled_img, strong=strong_unlabeled_img
                )
                weak_unlabeled_img = transformed["image"]
                strong_unlabeled_img = transformed["strong"]
            if self.unlabeled_strong_transform is not None:
                strong_unlabeled_img = self.unlabeled_strong_transform(
                    image=strong_unlabeled_img
                )["image"]
            weak_unlabeled_img = self.normalization(weak_unlabeled_img)
            strong_unlabeled_img = self.normalization(strong_unlabeled_img)

        return img, label_mask, weak_unlabeled_img, strong_unlabeled_img
