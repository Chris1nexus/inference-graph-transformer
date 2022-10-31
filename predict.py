
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import torchvision.models as models
from feature_extractor import cl
from models.GraphTransformer import Classifier
from models.weight_init import weight_init
from feature_extractor.build_graph_utils import ToTensor, Compose, bag_dataset, adj_matrix
import torchvision.transforms.functional as VF
from src.vis_graphcam import show_cam_on_image,cam_to_mask
from easydict import EasyDict as edict
from models.GraphTransformer import Classifier
from slide_tiling import save_tiles
import pickle
from collections import OrderedDict
import glob
import openslide
import numpy as np
import skimage.transform
import cv2


class Predictor:

    def __init__(self):
        self.classdict = pickle.load(open(os.environ['CLASS_METADATA'], 'rb' ))
        self.label_map_inv = dict()
        for label_name, label_id in self.classdict.items():
            self.label_map_inv[label_id] = label_name

        iclf_weights = os.environ['FEATURE_EXTRACTOR_WEIGHT_PATH']
        graph_transformer_weights = os.environ['GT_WEIGHT_PATH']
        self.__init_iclf(iclf_weights, backbone='resnet18')
        self.__init_graph_transformer(graph_transformer_weights)

    def predict(self, slide_path):

        # get tiles for a given WSI slide
        save_tiles(slide_path)

        filename = os.path.basename(slide_path)
        FILEID = filename.rsplit('.', maxsplit=1)[0]
        patches_glob_path = os.path.join(os.environ['PATCHES_DIR'], f'{FILEID}_files', '*', '*.jpeg')
        patches_paths = glob.glob(patches_glob_path)
        
        sample = self.iclf_predict(patches_paths)
        

        torch.set_grad_enabled(True)
        node_feat, adjs, masks = Predictor.preparefeatureLabel(sample['image'], sample['adj_s'])
        pred,labels,loss,graphcam_tensors = self.model.forward(node_feat=node_feat, labels=None, adj=adjs, mask=masks, graphcam_flag=True, to_file=False)
        
        patches_coords = sample['c_idx'][0]
        viz_dict = self.get_graphcams(graphcam_tensors, patches_coords, slide_path, FILEID)
        return self.label_map_inv[pred.item()], viz_dict

    def iclf_predict(self, patches_paths):
        feats_list = []

        batch_size = 128
        num_workers = 4
        args = edict({'batch_size':batch_size, 'num_workers':num_workers} )
        dataloader, bag_size = bag_dataset(args, patches_paths)

        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                feats, classes = self.i_classifier(patches)
                #feats = feats.cpu().numpy()
                feats_list.extend(feats)
        output = torch.stack(feats_list, dim=0).cuda()
        # save adjacent matrix
        adj_s = adj_matrix(patches_paths, output)      


        patch_infos = []
        for path in patches_paths:
            x, y = path.split('/')[-1].split('.')[0].split('_')
            patch_infos.append((x,y))

        preds = {'image': [output], 
                'adj_s': [adj_s],
                'c_idx': [patch_infos]}
        return preds        



    def get_graphcams(self, graphcam_tensors, patches_coords, slide_path, FILEID):
       label_map = self.classdict
       label_name_from_id = self.label_map_inv

       n_class = len(label_map)

       p =  graphcam_tensors['prob'].cpu().detach().numpy()[0]
       ori = openslide.OpenSlide(slide_path)
       width, height = ori.dimensions

       REDUCTION_FACTOR = 20
       w, h = int(width/512), int(height/512)
       w_r, h_r = int(width/20), int(height/20)
       resized_img = ori.get_thumbnail((width,height))#ori.get_thumbnail((w_r,h_r))
       resized_img = resized_img.resize((w_r,h_r))
       ratio_w, ratio_h = width/resized_img.width, height/resized_img.height
       #print('ratios ', ratio_w, ratio_h)
       w_s, h_s = float(512/REDUCTION_FACTOR), float(512/REDUCTION_FACTOR)

       patches = []
       xmax, ymax = 0, 0
       for patch_coords in patches_coords:
          x, y = patch_coords
          if xmax < int(x): xmax = int(x)
          if ymax < int(y): ymax = int(y)
          patches.append('{}_{}.jpeg'.format(x,y))



       output_img = np.asarray(resized_img)[:,:,::-1].copy()
       #-----------------------------------------------------------------------------------------------------#
       # GraphCAM
       #print('visulize GraphCAM')
       assign_matrix = graphcam_tensors['s_matrix_ori']
       m = nn.Softmax(dim=1)
       assign_matrix = m(assign_matrix)

       # Thresholding for better visualization
       p = np.clip(p, 0.4, 1)



       output_img_copy =np.copy(output_img)
       gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
       image_transformer_attribution = (output_img_copy - output_img_copy.min()) / (output_img_copy.max() - output_img_copy.min())
       cam_matrices = []
       masks = []
       visualizations = []
       viz_dict = dict()

       SAMPLE_VIZ_DIR = os.path.join(os.environ['GRAPHCAM_DIR'],
                            FILEID)
       os.makedirs(SAMPLE_VIZ_DIR, exist_ok=True)

       for class_i in range(n_class):

             # Load graphcam for each class
             cam_matrix = graphcam_tensors[f'cam_{class_i}']
             cam_matrix = torch.mm(assign_matrix, cam_matrix.transpose(1,0))
             cam_matrix = cam_matrix.cpu()

             # Normalize the graphcam
             cam_matrix = (cam_matrix - cam_matrix.min()) / (cam_matrix.max() - cam_matrix.min())
             cam_matrix = cam_matrix.detach().numpy()
             cam_matrix = p[class_i] * cam_matrix
             cam_matrix = np.clip(cam_matrix, 0, 1)      


             mask = cam_to_mask(gray, patches, cam_matrix, w, h, w_s, h_s)

             vis = show_cam_on_image(image_transformer_attribution, mask)
             vis =  np.uint8(255 * vis)

             cam_matrices.append(cam_matrix)
             masks.append(mask)
             visualizations.append(vis)
             viz_dict['{}'.format(label_name_from_id[class_i]) ] = vis
             cv2.imwrite(os.path.join(
                            SAMPLE_VIZ_DIR,
                            '{}_all_types_cam_{}.png'.format(FILEID, label_name_from_id[class_i] )
                            ), vis)
       h, w, _ = output_img.shape
       if h > w:
          vis_merge = cv2.hconcat([output_img] + visualizations)
       else:
          vis_merge = cv2.vconcat([output_img] + visualizations)


       cv2.imwrite(os.path.join(
                    SAMPLE_VIZ_DIR,
                    '{}_all_types_cam_all.png'.format(FILEID)), 
                    vis_merge)
       viz_dict['ALL'] = vis_merge 
       cv2.imwrite(os.path.join(
                        SAMPLE_VIZ_DIR,
                        '{}_all_types_ori.png'.format(FILEID )
                        ), 
                        output_img)
       viz_dict['ORI'] = output_img 
       return viz_dict






    def preparefeatureLabel(batch_graph, batch_adjs):
        batch_size = len(batch_graph)
        max_node_num = 0

        for i in range(batch_size):
            max_node_num = max(max_node_num, batch_graph[i].shape[0])
        
        masks = torch.zeros(batch_size, max_node_num)
        adjs =  torch.zeros(batch_size, max_node_num, max_node_num)
        batch_node_feat = torch.zeros(batch_size, max_node_num, 512)

        for i in range(batch_size):
            cur_node_num =  batch_graph[i].shape[0]
            #node attribute feature
            tmp_node_fea = batch_graph[i]
            batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

            #adjs
            adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]
            
            #masks
            masks[i,0:cur_node_num] = 1  

        node_feat = batch_node_feat.cuda()
        adjs = adjs.cuda()
        masks = masks.cuda()

        return node_feat, adjs, masks

    def __init_graph_transformer(self, graph_transformer_weights):
        n_class = len(self.classdict)
        model = Classifier(n_class)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(graph_transformer_weights))
        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model


    def __init_iclf(self, iclf_weights, backbone='resnet18'):
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
            num_feats = 512
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=False, norm_layer=nn.InstanceNorm2d)
            num_feats = 512
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
            num_feats = 2048
        if backbone == 'resnet101':
            resnet = models.resnet101(pretrained=False, norm_layer=nn.InstanceNorm2d)
            num_feats = 2048
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()
        i_classifier = cl.IClassifier(resnet, num_feats, output_class=2).cuda()
        
        # load feature extractor

        state_dict_weights = torch.load(iclf_weights)
        state_dict_init = i_classifier.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            if 'features' not in k:
                continue        
            name = k_0
            new_state_dict[name] = v
        i_classifier.load_state_dict(new_state_dict, strict=False)

        self.i_classifier = i_classifier





#0 load metadata dicitonary for class names
#1 TILE THE IMAGE
#2 FEED IT TO FEATURE EXTRACTOR
#3 PRODUCE GRAPH
#4  predict graphcams
import subprocess
import argparse
import os 
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--slide_path', type=str, help='path to the WSI slide')
    args = parser.parse_args()
    predictor = Predictor()

    predicted_class, viz_dict = predictor.predict(args.slide_path)
    print('Class prediction is: ', predicted_class)









