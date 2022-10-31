###########################################################################
# Created by: YI ZHENG
# Email: yizheng@bu.edu
# Copyright (c) 2020
###########################################################################

import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        parser.add_argument('--train_set', type=str, help='train')
        parser.add_argument('--val_set', type=str, help='validation')
        parser.add_argument('--model_path', type=str, help='path to trained model')
        parser.add_argument('--log_path', type=str, help='path to log files')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--train', action='store_true', default=False, help='train only')
        parser.add_argument('--test', action='store_true', default=False, help='test only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--log_interval_local', type=int, default=10, help='classification classes')
        parser.add_argument('--resume', type=str, default="", help='path for model')
        parser.add_argument('--graphcam', action='store_true', default=False, help='GraphCAM')
        parser.add_argument('--dataset_metadata_path', type=str, help='Location of the metadata associated with the created dataset: label mapping, splits and so on')


        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr

        args.num_epochs = 120
        args.lr = 1e-3             

        if args.test:
            args.num_epochs = 1
        return args
