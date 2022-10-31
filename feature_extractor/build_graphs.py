
from cl import IClassifier 
from build_graph_utils import *
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



def compute_feats(args, bags_list, i_classifier, save_path=None, whole_slide_path=None):
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        if  args.magnification == '20x':
            glob_path = os.path.join(bags_list[i], '*.jpeg')
            csv_file_path = glob.glob(glob_path)
            # line below was in the original version, commented due to errror with current version
            #file_name = bags_list[i].split('/')[-3].split('_')[0]
            
            file_name = glob_path.split('/')[-3].split('_')[0]
            
        if args.magnification == '5x' or args.magnification == '10x':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))

        dataloader, bag_size = bag_dataset(args, csv_file_path)
        print('{} files to be processed: {}'.format(len(csv_file_path), file_name))

        if os.path.isdir(os.path.join(save_path, 'simclr_files', file_name)) or len(csv_file_path) < 1:
            print('alreday exists')
            continue
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                feats, classes = i_classifier(patches)
                #feats = feats.cpu().numpy()
                feats_list.extend(feats)
        
        os.makedirs(os.path.join(save_path, 'simclr_files', file_name), exist_ok=True)

        txt_file = open(os.path.join(save_path, 'simclr_files', file_name, 'c_idx.txt'), "w+")
        save_coords(txt_file, csv_file_path)
        # save node features
        output = torch.stack(feats_list, dim=0).cuda()
        torch.save(output, os.path.join(save_path, 'simclr_files', file_name, 'features.pt'))
        # save adjacent matrix
        adj_s = adj_matrix(csv_file_path, output)
        torch.save(adj_s, os.path.join(save_path, 'simclr_files', file_name, 'adj_s.pt'))

        print('\r Computed: {}/{}'.format(i+1, num_bags))
        

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    parser.add_argument('--num_feats', default=512, type=int, help='Feature size')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--dataset', default=None, type=str, help='path to patches')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone')
    parser.add_argument('--magnification', default='20x', type=str, help='Magnification to compute features')
    parser.add_argument('--weights', default=None, type=str, help='path to the pretrained weights')
    parser.add_argument('--output', default=None, type=str, help='path to the output graph folder')
    args = parser.parse_args()
    
    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
    
    # load feature extractor
    if args.weights is None:
        print('No feature extractor')
        return
    state_dict_weights = torch.load(args.weights)
    state_dict_init = i_classifier.state_dict()
    new_state_dict = OrderedDict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        if 'features' not in k:
            continue        
        name = k_0
        new_state_dict[name] = v
    i_classifier.load_state_dict(new_state_dict, strict=False)
 
    os.makedirs(args.output, exist_ok=True)
    bags_list = glob.glob(args.dataset)
    print(bags_list)
    compute_feats(args, bags_list, i_classifier, args.output)
    
if __name__ == '__main__':
    main()
