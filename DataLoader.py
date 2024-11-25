import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import random
from albumentations.pytorch import ToTensorV2
import albumentations as A

class ImageDataset(Dataset):
    def __init__(self, path, name, target, aug):
        super(ImageDataset, self).__init__()
        self.path = path
        self.name = name
        self.target = target
        self.aug = aug

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        im_name = self.name[index]
        y = self.target[index]

        # Construct the image path and read the image
        img_path = os.path.join(self.path, im_name + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found.")

        img = cv2.resize(img, (384, 384))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Decide whether to apply augmentation
        apply_augmentation = y == 1 or random.random() < 0.00001
        
        if apply_augmentation:
            # Apply augmentations and normalize the augmented image
            augmented = self.aug(image=img)
            img_augmented = augmented['image']
        else:
            # Normalize and convert to tensor for non-augmented images
            img_normalized = img.astype(np.float32) / 255.0
            img_augmented = A.Compose([ToTensorV2()])(image=img_normalized)['image']

        # Convert the target value to a tensor
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return img_augmented, y_tensor


class Data_Loader():
    def __init__(self, path, name, target, aug):
        self.path = path
        self.name = name
        self.target = target
        self.aug = aug
        self.dataset = ImageDataset(self.path, self.name, self.target, self.aug)

    def get(self, batch_size, shuffle=True, num_workers=4):
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        return dataloader
