
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict

class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img 

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        img = img.resize((224, 224))
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def save_coords(txt_file, csv_file_path):
    for path in csv_file_path:
        x, y = path.split('/')[-1].split('.')[0].split('_')
        txt_file.writelines(str(x) + '\t' + str(y) + '\n')
    txt_file.close()

def adj_matrix(csv_file_path, output):
    total = len(csv_file_path)
    adj_s = np.zeros((total, total))

    for i in range(total-1):
        path_i = csv_file_path[i]
        x_i, y_i = path_i.split('/')[-1].split('.')[0].split('_')
        for j in range(i+1, total):
            # sptial 
            path_j = csv_file_path[j]
            x_j, y_j = path_j.split('/')[-1].split('.')[0].split('_')
            if abs(int(x_i)-int(x_j)) <=1 and abs(int(y_i)-int(y_j)) <= 1:
                adj_s[i][j] = 1
                adj_s[j][i] = 1

    adj_s = torch.from_numpy(adj_s)
    adj_s = adj_s.cuda()

    return adj_s

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)