import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config


class ForestSegmentationDataset(Dataset):
    """dataset"""

    def __init__(self, image_paths, mask_paths, transform=None, use_nrg=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.use_nrg = use_nrg

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_nrg and image.shape[2] == 4:  # assume that it has 4 channels
            image = image[:, :, [0, 1, 2, 3]]  # reserve all channels
        else:
            image = image[:, :, :3]  #  RGB 3 channels

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return image, mask


class DataManager:

    def __init__(self, config):
        self.config = config
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()

    def _get_train_transforms(self):
        """AUGMENTATION"""
        if self.config.USE_AUGMENTATION:
            return A.Compose([
                A.Resize(self.config.INPUT_SIZE[0], self.config.INPUT_SIZE[1]),
                A.Rotate(limit=self.config.ROTATION_RANGE, p=0.5),
                A.HorizontalFlip(p=0.5 if self.config.FLIP_HORIZONTAL else 0),
                A.VerticalFlip(p=0.5 if self.config.FLIP_VERTICAL else 0),
                A.RandomBrightnessContrast(
                    brightness_limit=self.config.BRIGHTNESS_RANGE,
                    contrast_limit=self.config.BRIGHTNESS_RANGE,
                    p=0.5
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return A.Compose([
                A.Resize(self.config.INPUT_SIZE[0], self.config.INPUT_SIZE[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _get_val_transforms(self):
        """transform"""
        return A.Compose([
            A.Resize(self.config.INPUT_SIZE[0], self.config.INPUT_SIZE[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_data_paths(self):
        img_dir = self.config.NRG_IMAGES_DIR if self.config.USE_NRG else self.config.RGB_IMAGES_DIR
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.png')]

        image_paths = []
        mask_paths = []

        for img_file in image_files:
            img_path = os.path.join(img_dir, img_file)

            if img_file.startswith("NRG_"):
                core_name = img_file[4:]
            else:
                core_name = img_file

            mask_file = f"mask_{core_name}"
            mask_path = os.path.join(self.config.MASKS_DIR, mask_file)

            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
            else:
                print(f"Warning: No matching mask for {img_file} at {mask_path}")

        print(f"\n Checking directory: {img_dir}")
        print(f"Found image files: {image_files[:5]}...")
        print(f" Mask directory: {self.config.MASKS_DIR}")
        print(f" Found mask files: {os.listdir(self.config.MASKS_DIR)[:5]}...")
        print(f"找到 {len(image_paths)} ")

        if len(image_paths) == 0:
            raise ValueError("❌ No files found")

        return image_paths, mask_paths

    def create_datasets(self):
        """create training and validation datasets"""
        image_paths, mask_paths = self.load_data_paths()

        # split data
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            image_paths, mask_paths,
            test_size=1 - self.config.TRAIN_SPLIT,
            random_state=self.config.RANDOM_SEED,
            stratify=None
        )

        # create dataset
        train_dataset = ForestSegmentationDataset(
            train_imgs, train_masks,
            transform=self.train_transform,
            use_nrg=self.config.USE_NRG
        )

        val_dataset = ForestSegmentationDataset(
            val_imgs, val_masks,
            transform=self.val_transform,
            use_nrg=self.config.USE_NRG
        )

        return train_dataset, val_dataset

    def create_dataloaders(self):
        """create training and validation dataloaders"""
        train_dataset, val_dataset = self.create_datasets()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader