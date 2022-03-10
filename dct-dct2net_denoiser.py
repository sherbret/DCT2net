#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : DCT2Net
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2021, v1.0.

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pickle
import argparse
from models.model_testing import *
from models.dct_denoiser import *
from skimage import feature
from skimage.morphology import binary_dilation

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, dest="model_name", help="Path to the saved model.", default="./saved_models/dct2net.p")
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Patch size.", default=13)
parser.add_argument("--sigma", type=float, dest="sigma", help="Standard deviation of the noise (noise level). Should be between 1 and 55.", default=25)
parser.add_argument("--in", type=str, dest="img_to_denoise", help="Path to the image to denoise.", default="./test_images/102061.png")
parser.add_argument("--out", type=str, dest="out_folder", help="Path to put the denoised image.", default="./denoised.png")
parser.add_argument("--add_noise", action='store_true', help="Add artificial Gaussian noise to the image.")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

im = Image.open(args.img_to_denoise)
img = np.array(im) 

if args.add_noise:
	img_noisy = img + args.sigma * np.random.randn(*img.shape)
else:
	img_noisy = img

img_noisy_torch = torch.from_numpy(img_noisy).view(1, 1, *img_noisy.shape).to(device).float() / 255


# Model
m = DCT2net(args.patch_size)
m.Pm1.data = torch.load(args.model_name, map_location=device).view(args.patch_size**2, args.patch_size**2)  
m.to(device)


m_dct = DCT_denoiser(16)  
m_dct.to(device)

# Denoising
with torch.no_grad():
	sigma_torch = args.sigma / 255 * torch.ones(1, 1, 1, device=device)
	
	img_denoised_torch1 = m_dct(img_noisy_torch, sigma_torch)
	img_denoised_torch2 = m(img_noisy_torch, sigma_torch)

	imgX2 = torch.cat((img_denoised_torch1, img_denoised_torch2), dim=1)

	edges = feature.canny(img_denoised_torch1.view(*img_noisy.shape).cpu().numpy(), sigma=1)
	
	d = binary_dilation(edges, footprint=np.ones((5,5)))
	d = torch.from_numpy(d).type(torch.int64).to(device)
	img_denoised_torch = torch.gather(imgX2, dim=1, index=d.view(1, 1, *img_noisy.shape))
	img_denoised = img_denoised_torch.view(*img_noisy.shape).cpu().numpy()
	img_denoised = np.clip(img_denoised, 0, 1) * 255

# Performance in PSNR
if args.add_noise:
	print("PSNR:", round(10*np.log10(255**2 / np.mean((img_denoised - img)**2)), 2), "dB")

# Saving
im = Image.fromarray(np.round(img_denoised).astype(np.uint8))
im.save(args.out_folder)






















