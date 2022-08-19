import torch
import cv2
import os
from torchvision import transforms
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn
from dataset import MyDataSet
from IoUs import  mIoU, Boundary_IoU
from models import *
import numpy as np

model = torch.load('best_UNet.pth')
#print(model)
data_transforms = {
    'rpg': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'map': transforms.Compose([
        transforms.ToTensor(),
    ])
}

image_datasets = {x:MyDataSet(
    root_dir='predict',
    train_val=x,
    transform=data_transforms
) for x in ['val']}

dataloaders = dict()

dataloaders['val'] = DataLoader(
    image_datasets['val'],
    batch_size=1,
    shuffle=False
)

data_sizes = {x:len(image_datasets[x]) for x in ['val']}

path ="results"
count = 0
for file in os.listdir(path): #count file number for creating new mask 
    count = count+1

for i, data in enumerate(dataloaders['val']):
    inputs , _ = data
    
    outputs = model(inputs)

    img_pred=torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    img_pred[img_pred>0] = 255
    
    cv2.imwrite('predict/results/pred_mask{}.jpg'.format(count),img_pred)
    count = count + 1

print("predict over")