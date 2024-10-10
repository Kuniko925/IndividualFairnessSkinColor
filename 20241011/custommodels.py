import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import Adamax
import torchvision.models as models


class TransDataset(Dataset):
    def __init__(self, dataframe, img_size, label, transform=None):
        self.dataframe = dataframe
        self.img_size = img_size
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    # idx　前から何番目か
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["filepath"]
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx][self.label]
        distance = [
            self.dataframe.iloc[idx]["WD"],
            self.dataframe.iloc[idx]["KUIPER"],
            self.dataframe.iloc[idx]["AD"],
            self.dataframe.iloc[idx]["CVM"],
            self.dataframe.iloc[idx]["KS"],
            self.dataframe.iloc[idx]["ED"]
        ]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(distance, dtype=torch.float32)

class LoadDataset:
    def __init__(self, label, batch_size):
        self.label = label
        self.batch_size = batch_size
        
    def create_dataloaders(self, df_train, df_valid, df_test):
        sample_image_path = df_train["filepath"].iloc[0]
        with Image.open(sample_image_path) as img:
            width, height = img.size
            print(f"Width: {width} Height: {height}")
            img_size = (height, width)
        
        label_encoder = LabelEncoder()
        df_train[self.label] = label_encoder.fit_transform(df_train[self.label])
        df_valid[self.label] = label_encoder.transform(df_valid[self.label])
        df_test[self.label] = label_encoder.transform(df_test[self.label])
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
        ])
        
        valid_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
    
        train_dataset = TransDataset(df_train, img_size, self.label, transform=train_transform)
        valid_dataset = TransDataset(df_valid, img_size, self.label, transform=valid_transform)
        test_dataset = TransDataset(df_test, img_size, self.label, transform=valid_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
        return train_loader, valid_loader, test_loader

class ResNet50(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.base_model.fc = nn.Linear(self.base_model.fc.weight.shape[1], num_class)
    def forward(self, x):
        x = self.base_model(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.base_model.classifier = nn.Linear(1024, num_class)
    def forward(self, x):
        x = self.base_model(x)
        return x

class MobileNetV2(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = self.base_model.classifier[1].in_features  # 最終Linear層の入力次元
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, num_class)
        )
    def forward(self, x):
        x = self.base_model(x)
        return x