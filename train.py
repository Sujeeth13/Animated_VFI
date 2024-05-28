import torch
import torch.nn as nn
from torch.nn import functional as F
from models.ResUNet import ResUNet
from models.TransUNet import TransUNet
from models.UNet import UNet
from models.OptNet import OptNet
from utils.data_loader import create_loaders
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import datetime
import numpy as np
from piqa import SSIM, PSNR
import wandb
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        self.feature_extractor1 = nn.Sequential(*list(vgg.children())[:16]).to(device)
        self.feature_extractor2 = nn.Sequential(*list(vgg.children())[:8]).to(device)
        self.feature_extractor3 = nn.Sequential(*list(vgg.children())[:12]).to(device)
        
        for param in self.feature_extractor1.parameters():
            param.requires_grad = False
        for param in self.feature_extractor2.parameters():
            param.requires_grad = False
        for param in self.feature_extractor3.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        ploss1 = nn.MSELoss()(self.feature_extractor1(input), self.feature_extractor1(target))
        ploss2 = nn.MSELoss()(self.feature_extractor2(input), self.feature_extractor2(target))
        ploss3 = nn.MSELoss()(self.feature_extractor3(input), self.feature_extractor3(target))
        return ploss1 + ploss2 + ploss3


def dataload(file_path,batch_size,n_w,task):
    train_loader, test_loader = create_loaders(file_path, task, batch_size=batch_size, test_size=0.2, 
                                                random_seed=42, n_w=n_w)
    return train_loader, test_loader

def setup(lr,wd,in_channels,out_channels,n_layers=5,bn_layers=2,model_path=None,model_type=1,task='vfi'):
    assert len(in_channels) == n_layers and len(out_channels) == n_layers, \
    'Error: channels should be same as number of layers'
    if task == 'vfi':
        if model_type == 0:
            print("Using simple UNet")
            model = UNet(in_channels=in_channels,out_channels=out_channels,
                            blocks=n_layers,bn_blocks=bn_layers)
        elif model_type == 1:
            print("Using ResUNet")
            model = ResUNet(in_channels=in_channels,out_channels=out_channels,
                    blocks=n_layers,bn_blocks=bn_layers)
        else:
            print("Using TransUNet")
            model = TransUNet(in_channels=in_channels,out_channels=out_channels,
                    blocks=n_layers,bn_blocks=bn_layers)
    elif task == 'optFlow':
        print("Using OptNet")
        model = OptNet(in_channels=in_channels,out_channels=out_channels,
                            blocks=n_layers,bn_blocks=bn_layers)
    else:
        raise Exception("Invalid task, use --task\"vfi\" or --task\"optFlow\"")
        
        
    if model_path is not None:
        try:
            print("Loading model")
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print("Couldn't load model weights")
            print("Error:",e)

    optim = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
    
    if task == 'vfi':
        criteria = [PerceptualLoss(device='cuda'), nn.L1Loss()]
    elif task == 'optFlow':
        criteria = [nn.MSELoss()]
        
    return model,criteria,optim

def calculate_ssim_psnr(img1, img2):
    # Initialize metrics
    ssim = SSIM().cuda()  # Move SSIM computation to GPU
    psnr = PSNR().cuda() 

    ssim_value = ssim(img1, img2)
    psnr_value = psnr(img1, img2)

    return ssim_value.detach(), psnr_value.detach()

def train_vfi(data_loader,test_loader,model,epochs,device,criteria,optim,local_rank,rank):
    print(f"Proc {rank} using device {device}")
    model = DDP(model,device_ids=[local_rank])
    total_step = len(data_loader)
    best_test_ssim = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 

    for epoch in range(epochs):
        avg_train_loss = 0
        avg_test_loss = 0
        avg_train_ssim = 0
        avg_train_psnr = 0
        avg_test_ssim = 0
        avg_test_psnr = 0
        total_train_batch = 0
        total_test_batch = 0
        lamda = 1.2
        
        data_loader.sampler.set_epoch(epoch)

        print(f"Epoch {epoch + 1}:")
        data_iterator = iter(data_loader)

        for _ in tqdm(range(len(data_loader)),desc='Training',disable=(rank != 0)):
            model.train()
            images,labels = next(data_iterator)
            mask = images[2].to(device)
            images = images[:-1]
            # Move tensors to configured device
            images = torch.cat(images,dim=1)
            images = images.to(device)
            labels = labels.to(device)
            optim.zero_grad()
            

            # Forward pass
            outputs = model(images,mask)
            loss = criteria[0](outputs, labels) + lamda*criteria[1](mask*outputs, mask*labels)

            # Backward and optimize
            loss.backward()
            optim.step()

            model.eval()
            # Calculate accuracy and loss
            avg_train_loss += loss.item()

            train_ssim , train_psnr = calculate_ssim_psnr(outputs, labels)
            avg_train_ssim += train_ssim
            avg_train_psnr += train_psnr

            total_train_batch += 1

        test_iterator = iter(test_loader)
        for _ in  tqdm(range(len(test_loader)),desc='Testing'):
            model.eval()
            images,labels = next(test_iterator)
            mask = images[2].to(device)
            images = images[:-1]
            # Move tensors to configured device
            images = torch.cat(images,dim=1)
            images = images.to(device)
            labels = labels.to(device)

            # Calculate accuracy
            outputs = model(images,mask)
            loss = criteria[0](outputs, labels) + lamda*criteria[1](mask*outputs, mask*labels)

            avg_test_loss += loss.item()
            
            test_ssim , test_psnr = calculate_ssim_psnr(outputs, labels)
            avg_test_ssim += test_ssim
            avg_test_psnr += test_psnr

            total_test_batch += 1

        avg_train_ssim = avg_train_ssim/total_train_batch
        avg_train_psnr = avg_train_psnr/total_train_batch
        avg_train_loss = avg_train_loss/total_train_batch
        avg_test_ssim = avg_test_ssim/total_test_batch
        avg_test_psnr = avg_test_psnr/total_test_batch
        avg_test_loss = avg_test_loss/total_test_batch
        
        if rank == 0:
            print(
                f'Proc: {rank} Epoch [{epoch+1}/{epochs}], \
                Training SSIM: {avg_train_ssim:.4f}, \
                Training PSNR: {avg_train_psnr:.4f}, \
                Training Loss: {avg_train_loss:.4f}, \
                Test SSIM: {avg_test_ssim:.4f}, \
                Test PSNR: {avg_test_psnr:.4f}, \
                Test Loss: {avg_test_loss:.4f}'
            )

            wandb.log({
                "Average Training Loss": avg_train_loss,
                "Average Training SSIM": avg_train_ssim,
                 "Average Training PSNR": avg_train_psnr,
                "Average Test Loss": avg_test_loss,
                "Average Test SSIM": avg_test_ssim,
                "Average Test PSNR": avg_test_psnr,
                "epoch": epoch
            })


        if avg_test_ssim > best_test_ssim and rank == 0:
            best_test_ssim = avg_test_ssim
            torch.save(model.module.state_dict(), f'best_model_{timestamp}.pth')
            print(f'Best model saved with Test SSIM: {avg_test_ssim:.4f}')

def train_optFlow(data_loader,test_loader,model,epochs,device,criteria,optim,local_rank,rank):
    print(f"Proc {rank} using device {device}")
    model = DDP(model,device_ids=[local_rank])
    best_test_loss = 1000000
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 

    for epoch in range(epochs):
        avg_train_loss = 0
        avg_test_loss = 0
        total_train_batch = 0
        total_test_batch = 0
        
        data_loader.sampler.set_epoch(epoch)

        print(f"Epoch {epoch + 1}:")
        data_iterator = iter(data_loader)

        for _ in tqdm(range(len(data_loader)),desc='Training',disable=(rank != 0)):
            model.train()
            images,labels = next(data_iterator)
            images = torch.cat(images,dim=1)
            images = images.to(device)
            labels = labels.to(device)
            optim.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criteria[0](outputs, labels)

            # Backward and optimize
            loss.backward()
            optim.step()

            model.eval()
            # Calculate accuracy and loss
            avg_train_loss += loss.item()

            total_train_batch += 1

        test_iterator = iter(test_loader)
        for _ in  tqdm(range(len(test_loader)),desc='Testing'):
            model.eval()
            images,labels = next(test_iterator)
            
            # Move tensors to configured device
            images = torch.cat(images,dim=1)
            images = images.to(device)
            labels = labels.to(device)

            # Calculate accuracy
            outputs = model(images)
            loss = criteria[0](outputs, labels)

            avg_test_loss += loss.item()

            total_test_batch += 1

        avg_train_loss = avg_train_loss/total_train_batch
        avg_test_loss = avg_test_loss/total_test_batch
        
        if rank == 0:
            print(
                f'Proc: {rank} Epoch [{epoch+1}/{epochs}], \
                Training Loss: {avg_train_loss:.4f}, \
                Test Loss: {avg_test_loss:.4f}'
            )

            wandb.log({
                "Average Training Loss": avg_train_loss,
                "Average Test Loss": avg_test_loss,
                "epoch": epoch
            })


        if avg_test_loss < best_test_loss and rank == 0:
            best_test_loss = avg_test_loss
            torch.save(model.module.state_dict(), f'optFlow_model_{timestamp}.pth')
            print(f'Best model saved with Test Loss: {avg_test_loss:.4f}')
            
def train(data_loader,test_loader,model,epochs,device,criteria,optim,local_rank,rank,task):
    if task == 'vfi':
        train_vfi(data_loader,test_loader,model,epochs,device,criteria,optim,local_rank,rank)
    elif task == 'optFlow':
        train_optFlow(data_loader,test_loader,model,epochs,device,criteria,optim,local_rank,rank)
    else:
        raise Exception("Invalid task")
