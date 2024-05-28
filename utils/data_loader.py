import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.distributed import DistributedSampler

class ImageDataset(Dataset):
    def __init__(self, file_path, indices, task='vfi', mode='train'):
        self.file = h5py.File(file_path, 'r')
        self.indices = indices
        self.mode = mode
        self.task = task

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        group_key = list(self.file)[self.indices[idx]]
        data = self.file[group_key]
        
        if self.task == 'vfi':
        
            input_frame1 = torch.tensor(data['frame1'][:], dtype=torch.float32).permute(2,0,1) / 255.0
        
            mask = np.mean(data['frame1'][:], -1) - np.mean(data['frame3'][:], -1)
            mask = np.expand_dims(np.abs(mask), -1)
            mask = torch.tensor(mask).permute(2,0,1) / 255.0
            
            if self.mode == 'train':
                input_flow = torch.tensor(data['flow13'][:], dtype=torch.float32)
                output_frame = torch.tensor(data['frame3'][:], dtype=torch.float32).permute(2,0,1) / 255.0 
                        
            elif self.mode == 'test':
                flow13 = torch.tensor(data['flow13'][:], dtype=torch.float32)
                flow31 = torch.tensor(data['flow31'][:], dtype=torch.float32)
                input_flow = (flow13 + flow31)*0.5
                
                output_frame = torch.tensor(data['frame2'][:], dtype=torch.float32).permute(2,0,1) / 255.0            
            else:
                raise Exception('Invalid mode')
            
            return (input_frame1, input_flow, mask), output_frame

        elif self.task == 'optFlow':
            
            input_frame1 = torch.tensor(data['frame1'][:], dtype=torch.float32).permute(2,0,1) / 255.0
            input_frame3 = torch.tensor(data['frame3'][:], dtype=torch.float32).permute(2,0,1) / 255.0
            flow13 = torch.tensor(data['flow13'][:], dtype=torch.float32)
            flow31 = torch.tensor(data['flow31'][:], dtype=torch.float32)
            
            return (input_frame1, input_frame3), torch.cat((flow13, flow31), dim=0)
        else:
            raise Exception('Invalid task')

    def close(self):
        self.file.close()

def create_loaders(file_path, task, batch_size=32, test_size=0.2, random_seed=42, n_w=2):
    with h5py.File(file_path, 'r') as file:
        total_images = len(file)
        print(total_images)
        indices = list(range(total_images))
    
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_seed)
    
    train_dataset = ImageDataset(file_path, train_indices, task, mode='train')
    test_dataset = ImageDataset(file_path, test_indices, task, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                              sampler=DistributedSampler(train_dataset,shuffle=True), num_workers=n_w)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_test_loader(file_path, task, batch_size=32, test_size=0.2, random_seed=42, n_w=2):
    with h5py.File(file_path, 'r') as file:
        total_images = len(file)
        indices = list(range(total_images))
    
    _, test_indices = train_test_split(indices, test_size=test_size, random_state=random_seed)
    
    test_dataset = ImageDataset(file_path, test_indices, task, mode='train')
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

if __name__ == '__main__':
    file_path = '../image_data.h5'
    train_loader, test_loader = create_loaders(file_path, batch_size=64)

    for (inputs_gray, inputs_mask, inputs_user_img), (targets_a, targets_b) in train_loader:
        print(inputs_gray.shape, inputs_mask.shape, inputs_user_img.shape, targets_a.shape, targets_b.shape)