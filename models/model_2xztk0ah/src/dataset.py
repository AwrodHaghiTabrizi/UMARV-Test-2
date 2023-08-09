import sys
import numpy as np
import cv2
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import random

class Dataset_Class(Dataset):

  def __init__(self, data_dirs, label_dirs, device, label_input_threshold=.1):
    self.data_dirs = data_dirs
    self.label_dirs = label_dirs
    self.device = device
    self.input_threshold = .1

    self.requisite_data_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((224, 224), antialias=None)
    ])
    self.requisite_label_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((224, 224), antialias=None),
      transforms.Grayscale(1)
    ])
    self.default_data_transform = transforms.Compose([
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)


  def __len__(self):
      return len(self.data_dirs)

  def __getitem__(self, idx):
    data = cv2.imread(self.data_dirs[idx], cv2.IMREAD_COLOR)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = self.requisite_data_transform(data)
    data = data.to(self.device)
    data_raw = data.detach().clone()
    data = self.default_data_transform(data)
    label = cv2.imread(self.label_dirs[idx])
    label = self.requisite_label_transform(label)
    label = label.to(self.device)
    label_0 = torch.zeros(label.shape, device=self.device)
    label_0[label < self.input_threshold] = 1
    label_1 = torch.zeros(label.shape, device=self.device)
    label_1[label >= self.input_threshold] = 1
    label = torch.stack((label_0.squeeze(), label_1.squeeze()))
    label = label.to(self.device)
    if random.random() > .5:
      data_raw = self.flip_transform(data_raw)
      data = self.flip_transform(data)
      label = self.flip_transform(label)
    return data_raw, data, label