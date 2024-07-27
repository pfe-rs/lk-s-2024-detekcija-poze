import os
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, heat_maps_dir, img_dir, transform, heat_maps_input, target_transform=None):
        self.heat_maps_dir = heat_maps_dir
        self.img_dir = img_dir
        
        self.heat_maps_input = heat_maps_input
        self.transform = transform = transforms.Compose([
            transforms.Resize(transform),
            ])
        self.target_transform = None
        #self.center_map = center_map

    def __len__(self):
        return len([name for name in os.listdir(self.img_dir)])

    def __getitem__(self, idx):
        img_files = os.listdir(self.img_dir)
        heatmap_files = os.listdir(self.heat_maps_dir)
        heat_maps_input_files = os.listdir(self.heat_maps_input)
        
        img_name = img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = read_image(img_path).float() / 255.0
        
            

        heatmap_name = f'hm_{img_name.replace(".jpg", ".npy")}'
        heatmap_input_name = f'hmi_{img_name.replace(".jpg", ".npy")}'

        if heatmap_name not in heatmap_files:
            raise FileNotFoundError(f"Heatmap {heatmap_name} not found for image {img_name}")
        heatmap_path = os.path.join(self.heat_maps_dir, heatmap_name)
        heatmap = np.load(heatmap_path, allow_pickle=True)


        if heatmap_input_name not in heat_maps_input_files:
            raise FileNotFoundError(f"Heatmap in {heatmap_name} not found for image {img_name}")
        heatmap_input_path = os.path.join(self.heat_maps_input, heatmap_input_name)
        heatmap_input = np.load(heatmap_input_path, allow_pickle=True)
        heatmap_input = torch.from_numpy(np.array([heatmap_input]))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            heatmap = self.target_transform(heatmap)
        
        
        heatmap = torch.tensor(heatmap)

        

        return image, heatmap, heatmap_input

    
class DataLoader1():
    def __init__(self, image_dataset, batch_size):
        self.used = []
        self.image_dataset = image_dataset
        self.batch_size = batch_size
        self.batch = []

#     def vrati(self, i):
#         self.batch = []
#         for x in os.listdir(self.image_dataset.img_dir):
#             if x not in self.used:
#                 idx = self.batch_size * i + len(self.batch)
#                 batch1 = self.image_dataset[idx]
#                 self.batch.append(batch1)
#                 self.used.append(x)

#             if len(self.batch) >= self.batch_size:
#                 y = self.batch
#                 return y