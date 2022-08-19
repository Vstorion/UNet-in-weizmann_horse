import torch
import time
import os
import argparse
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn
from dataset import MyDataSet
from models import *


def dice_loss(prediction, target):
    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def train_model(model, data_sizes, num_epochs, scheduler, dataloaders,criterion, optimizer, batch_size=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device1 = 'cpu'
    since = time.time()
    dir_num=0
    for epoch in range(num_epochs):
        begin_time = time.time()
        running_loss = 0.0
        count_batch = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('*'*40)
    
        for phase in ['train']:
            
            scheduler.step()
            model.train(True)
            for i, data in enumerate(dataloaders[phase]):
                count_batch += 1
                inputs, labels = data
                inputs, labels = Variable(inputs).to(device), Variable(labels).squeeze().unsqueeze(0).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())\
                           + dice_loss(torch.argmax(outputs, dim=1).float(),
                                       labels.long().float())
                loss.backward()
                optimizer.step()

        ######### interact display area ###############
        
                running_loss += loss.data
                batch_loss = running_loss / (batch_size * count_batch)
                print('{} Epoch [{}] Batch [{}] Batch Loss: {:.14f} Time: {:.4f}s'.format(
                    phase, epoch, i, batch_loss, time.time()-begin_time
                ))
                begin_time = time.time()
                
        epoch_loss = running_loss / data_sizes[phase]
        print('{} Loss: {:.14f}'.format(phase, epoch_loss))

#         if (epoch+1)%5==0 :
#             if not os.path.exists(model_path):
#                 os.mkdir(model_path)
#             torch.save(model.cpu(), os.path.join(model_path, 'best_net{}.pth').format(dir_num))
#             dir_num=dir_num+1

        time_elapsed = time.time() - since
        print('Training completed in {:.0f}mins {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    return(model)


if __name__ == '__main__':
#################### initialize ##########################
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type = int)
    parser.add_argument("--modelsave_path", default='weight', type=str)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--step_size', default=2, type=int)
    parser.add_argument('--pretrained', default=None, type=int)
    args = parser.parse_args()
    
    use_gpu = torch.cuda.is_available()
    batch_size = args.batch_size
    modelsave_path = args.modelsave_path
    num_epochs = args.num_epochs
    if not os.path.exists(modelsave_path):
        os.mkdir(modelsave_path)
    
    if not use_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')
    
    device = torch.device("cuda:0" if use_gpu else "cpu")
    
    ########################## data process ##########################
    
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
        train_val=x,
        transform=data_transforms
    ) for x in ['train']}
    
    dataloaders = dict()
    
    dataloaders['train'] = DataLoader(
        image_datasets['train'],
        batch_size=batch_size,
        shuffle=True
    )
    
    data_sizes = {x:len(image_datasets[x]) for x in ['train']}
    
    ################### train ###################
    model = torch.load(args.pretrained) if args.pretrained else UNet(n_channels=3, n_classes=2)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.6)
    model = train_model(model=model,
                        data_sizes = data_sizes,
                        dataloaders=dataloaders,
                        num_epochs = num_epochs,
                        scheduler=exp_lr_scheduler,
                        criterion=criterion,
                        optimizer= optimizer)
    
    torch.save(model.cpu(), os.path.join(modelsave_path, 'best_UNet.pth2'))    