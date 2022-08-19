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
    root_dir='demo',
    train_val=x,
    transform=data_transforms
) for x in ['demo']}

dataloaders = dict()
dataloaders['demo'] = DataLoader(
    image_datasets['demo'],
    batch_size=1,
    shuffle=False
)

data_sizes = {x:len(image_datasets[x]) for x in ['demo']}

mIoUs = 0
BIoUs = 0

path ="demo/results"
count = 0
for file in os.listdir(path): #count file number for creating new mask 
    count = count+1

for i, data in enumerate(dataloaders['demo']):
    inputs, labels = data
    
    outputs = model(inputs)

    img_label=(labels.detach().numpy()).squeeze()
    img_pred=torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    
    miou = mIoU(img_label,img_pred)
    biou = Boundary_IoU(img_label.astype(np.uint8),img_pred.astype(np.uint8))
    mIoUs = mIoUs + miou
    BIoUs = BIoUs + biou
    print(f"The No.{i} Test Image's mIoU is{miou} ")
    print(f"The No.{i} Test Image's BIoU is{biou} ")
    img_label[img_label>0] = 255
    img_pred[img_pred>0] = 255
    
    cv2.imwrite('demo/results/pred_mask{}.jpg'.format(count),img_pred)
    count = count + 1

print(f"The average mIoU is{mIoUs/data_sizes['demo']} ")
print(f"The average BIoU is{BIoUs/data_sizes['demo']} ")
