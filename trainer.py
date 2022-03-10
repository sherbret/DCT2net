#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : DCT2Net
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2021, v1.0.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import MultiStepLR
import os
from os import listdir
from os.path import isfile, join
import pickle
import argparse
from models.model_training import DCT2net

parser = argparse.ArgumentParser()

# model
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Size of the patches.", default=13)

# training
parser.add_argument("--sigma_low", type=float, dest="sigma_low", help="Model is trained for all noise levels between sigma_low and sigma_high.", default=1)
parser.add_argument("--sigma_high", type=float, dest="sigma_high", help="Model is trained for all noise levels between sigma_low and sigma_high.", default=55)
parser.add_argument("--lr", type=float, dest="lr", help="ADAM learning rate.", default=0.001)
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs for training.", default=15)
parser.add_argument("--batch_size", type=int, dest="batch_size", default=32, help='Batch size during training.')
parser.add_argument("--patch_size_training", type=int, dest="patch_size_training", default=128, help='Training is done on small crops of images of size pxp.')
parser.add_argument("--stride_training", type=int, dest="stride_training", default=10, help='Training is done on small crops of images of size pxp.')

# dataset
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default="dct2net")
parser.add_argument("--in_folder", type=str, dest="in_folder", help="Path to the folder containing the images for the training.")
parser.add_argument("--out_folder", type=str, dest="out_folder", help="Path to the folder where models are saved.", default="./saved_models")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DenoisingDataset(Dataset):

    def __init__(self, in_folder, patch_size_training, stride_training, sigma_low, sigma_high):
        self.in_folder = in_folder
        self.patch_size_training = patch_size_training
        self.stride_training = stride_training
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high

        self.images_name = sorted([f for f in listdir(in_folder) if isfile(join(in_folder, f)) and (f[-3:] == "jpg" or f[-3:] == "png")])
        h, w = np.array(ImageOps.grayscale(Image.open(self.in_folder + "/" + self.images_name[0]))).shape # assuming that all images have the same dimensions
        self.n_patches = len(self.images_name) * ((h - patch_size_training) // stride_training) * ((w - patch_size_training) // stride_training)
        self.nb_patch_per_img = self.n_patches // len(self.images_name)

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        img_name = self.in_folder + "/" + self.images_name[idx // self.nb_patch_per_img]
        img_np = np.array(ImageOps.grayscale(Image.open(img_name))) / 255

        h, w = img_np.shape
        nb_patch_h = (h - self.patch_size_training) // self.stride_training
        nb_patch_w = (w - self.patch_size_training) // self.stride_training
        idx = idx % self.nb_patch_per_img

        i, j = idx // nb_patch_w, idx % nb_patch_w
        patch = img_np[i*self.stride_training:i*self.stride_training+self.patch_size_training, j*self.stride_training:j*self.stride_training+self.patch_size_training]
        patch = patch.astype(np.float32)

        sigma = np.random.uniform(low=self.sigma_low/255, high=self.sigma_high/255)
        patch_noisy = patch + sigma * np.random.randn(*patch.shape)
        patch_noisy = patch_noisy.astype(np.float32)

        # Data augmentation
        k1, k2 = np.random.randint(4), np.random.randint(3)
        patch = np.rot90(patch, k1)
        patch_noisy = np.rot90(patch_noisy, k1)
        if k2 < 2:
            patch = np.flip(patch, k2)
            patch_noisy = np.flip(patch_noisy, k2)
            
        patch = np.ascontiguousarray(patch)
        patch_noisy = np.ascontiguousarray(patch_noisy)
       
        img_torch = torch.from_numpy(patch).view(1, *patch.shape).to(device)
        img_noisy_torch = torch.from_numpy(patch_noisy).view(1, *patch.shape).to(device)
        sigma_torch = sigma * torch.ones(1, 1, 1, device=device)
        
        return 2*img_torch-1, 2*img_noisy_torch-1, 2*sigma_torch


dataset = DenoisingDataset(args.in_folder, args.patch_size_training, args.stride_training, args.sigma_low, args.sigma_high)

torch.manual_seed(99) # for reproductibility 
np.random.seed(99) # for reproductibility  

m = DCT2net(patch_size=args.patch_size)
m.to(device)

optimizer = optim.Adam(m.parameters(), lr=args.lr)
scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
loss_function = nn.MSELoss()

for epoch in range(1, args.num_epochs+1):
    data_loader_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    for i, elt in enumerate(data_loader_train):
        img_torch, img_noisy_torch, sigma_torch = elt
        q = args.patch_size - 1
        img_torch = img_torch[:, :, q:-q, q:-q]

        optimizer.zero_grad()
        loss = loss_function(m(img_noisy_torch, sigma_torch), img_torch)
        loss.backward()
        optimizer.step()
    scheduler.step()
    torch.save(m.state_dict()["dct.weight"], args.out_folder + "/" + args.model_name + "_" + str(epoch) + ".p")
