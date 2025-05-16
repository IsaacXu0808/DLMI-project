import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import csv
import numpy as np

import random
import numpy as np

def set_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, size=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            mask_transform (callable, optional): Optional transform to be applied on a mask.
            ratio (float): Ratio of images to be used for training. Default is 1.
        """
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self._transform = transform

        self.image_ids = [f[:-4] for f in os.listdir(image_dir) if f.endswith(".jpg")]
        if size:
            assert size <= len(self.image_ids), "Selected size {i} is larger than the training daraset.".format(i=size)
            self.size = size
            # if size == 1:
            #     self.image_ids = ["0014966"]
            # else:
            self.image_ids = random.sample(self.image_ids, self.size)
        else:
            self.size = len(self.image_ids)
        self._images = np.array([Image.open(os.path.join(self.image_dir, img_id + ".jpg")).convert("RGB") for img_id in self.image_ids])
        self._masks = np.array([Image.open(os.path.join(self.mask_dir, img_id + ".png")).convert("L") for img_id in self.image_ids])
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = self._images[idx % len(self.image_ids)]
        mask = self._masks[idx % len(self.image_ids)]

        image = image.transpose([2,0,1])  # transpose dimensions such that image shape is: channels, height, width
        image = image.astype(np.float32) / 255  # convert image from 8-bit integer to 32-bit floating precision
        mask = mask.astype(np.float32) / 255
        image = torch.as_tensor(image.copy())  # cast NumPy array to Torch tensor
        mask = torch.as_tensor(mask.copy()).unsqueeze(0)
        if self._transform is not None:
            combined = torch.cat([image, mask], dim=0)
            combined = self._transform(combined)
            image2 = combined[:len(image)]
            mask2 = combined[len(image):]
            image, mask = image2, mask2

        return image, mask

class ClassificationDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, ratio=1):
        assert ratio <= 1 and ratio > 0 , "Ratio must be between 0 and 1"
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        c_data = [[] for _ in range(3)]
        self.label_names = ["other"]
        with open(os.path.join(csv_path), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.label_names += header[1:]

            for row in reader:
                image_id = row[0].split('_')[1]
                mel = int(float(row[1]))
                sk = int(float(row[2]))

                if mel == 1:
                    label = 1
                elif sk == 1:
                    label = 2
                else:
                    label = 0

                c_data[label].append(
                    (np.array(Image.open(os.path.join(self.image_dir, image_id + ".jpg")).convert("RGB")), 
                     label)
                )
        self.data = c_data[0][:round(len(c_data[0]) * ratio)] + \
                    c_data[1][:round(len(c_data[1]) * ratio)] + \
                    c_data[2][:round(len(c_data[2]) * ratio)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = image.transpose([2,0,1])
        image = image.astype(np.float32) / 255
        image = torch.from_numpy(image).float()
        label = torch.tensor(label).float()

        if self.transform:
            image = self.transform(image)

        return image, label
    
def add_gaussian_noise(image, mean=0.0, std=0.05):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)

class ReconstructionDataset(Dataset):
    def __init__(self, image_dir, transform=None, mean=0.0, std=0.05):
        super().__init__()
        self.mean = mean
        self.std = std
        self.image_dir = image_dir
        self._images = np.array([np.array(Image.open(os.path.join(self.image_dir, file_name)).convert("RGB")) 
                                 for file_name in os.listdir(image_dir)])
        self.transform = transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx % len(self._images)]
        image = image.transpose([2,0,1])
        image = image.astype(np.float32) / 255
        image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        masked_image = add_gaussian_noise(image, mean=self.mean, std=self.std)

        return masked_image, image 
