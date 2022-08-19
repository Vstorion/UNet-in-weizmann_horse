from torch.utils.data import  Dataset
import os
import glob
from PIL import Image
import torch.optim as optim
import numpy as np

class MyDataSet(Dataset):

    def __init__(self, root_dir='train', \
                  train_val='train', transform=None):
        
        self.data_path = os.path.join(root_dir, 'horse')
        self.map_path = os.path.join(root_dir, 'mask')
        self.image_names = glob.glob(self.data_path + '/*.png')
        self.map_names = glob.glob(self.map_path + '/*.png')
        self.data_transform = transform
        self.train_val = train_val
        #print(self.image_names[0])

    def __len__(self):

        return len(self.image_names)

    def __getitem__(self, item):
        
        img_path = self.image_names[item]
        img = Image.open(img_path)
        image = img
        
        if self.train_val != 'val': 
            map_path = self.map_names[item]
            maps = np.array(Image.open(map_path))
            maps = Image.fromarray(maps.astype(np.float32))
            image_map = maps

        if self.data_transform is not None:
            try:
                image = self.data_path
                image = self.data_transform['rpg'](img)
                
            except:
                print('can not load image:{}'.format(img_path))
                
            try:
                image_map = self.map_path
                image_map = self.data_transform['map'](maps)
            except:
                pass
                
        
        return image, image_map
    
    def getitem(self, item):

        img_path = self.image_names[item]
        img = Image.open(img_path)
        image = img
        
        map_path = self.map_names[item]
        maps = Image.open(map_path)
        image_map = maps
        
        return image, image_map 
