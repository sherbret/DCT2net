#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : DCT2Net
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2021, v1.0.

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DCT2net(nn.Module):
    """ Unfold version of DCT2net"""
    def __init__(self, patch_size=13):
        super(DCT2net, self).__init__()

        self.patch_size = patch_size
        
        self.Pm1 = nn.Parameter(torch.randn(patch_size**2, patch_size**2, device=device))

        self.threshold = nn.Hardshrink(1.0)
        self.unfold = nn.Unfold(kernel_size=patch_size)

    def forward(self, x, sigma_):
        p = self.patch_size
        x = 2*x - 1
        sigma = sigma_ * 2
        x = torch.nn.functional.pad(x, (p-1, p-1, p-1, p-1), mode='reflect')
        N, _, H, W = x.size()
        
        x = self.unfold(x)
        x = self.Pm1 @ x
        
        lambda_ = 3*sigma
        
        x = x / lambda_
        x = self.threshold(x)
        x = x * lambda_
        
        count_nz = torch.count_nonzero(x[:, 1:, :], dim=1)
        w = 1/(1+count_nz)
        x = torch.inverse(self.Pm1) @ x
        
        x = w * x
        x = torch.nn.functional.fold(x, output_size=(H, W), kernel_size=p)
        
        ones_patches = torch.ones(N, p**2, (H-p+1) * (W-p+1), device=device)
        divisor = torch.nn.functional.fold(w * ones_patches, output_size=(H, W), kernel_size=p)
        x = x / divisor
        x = x[:, :, p-1:-(p-1), p-1:-(p-1)]
        return (x + 1)/2

# class DCT2net(nn.Module):
#     """ Convolution version of DCT2net (equivalent to the unfold one) """
#     def __init__(self, patch_size=13):
#         super(DCT2net, self).__init__()

#         self.patch_size = patch_size

#         my_filter = torch.zeros(patch_size**2, 1, patch_size, patch_size)
#         for k in range(patch_size**2):
#             i, j = k // patch_size, k % patch_size
#             my_filter[k, 0, i, j] = 1.0

#         self.fold = nn.ConvTranspose2d(patch_size**2, 1, patch_size, bias=False)
#         self.fold.weight = nn.Parameter(my_filter)
#         self.fold.requires_grad_(False)

#         self.small_conv = nn.Conv2d(1, 1, patch_size, bias=False)
#         self.small_conv.weight = nn.Parameter(torch.ones(1, 1, patch_size, patch_size))
#         self.small_conv.requires_grad_(False)


#         self.dct = nn.Conv2d(1, patch_size**2, patch_size, bias=False, padding=patch_size-1, padding_mode='reflect')

#         self.udiff = myUDiff.apply


#     def forward(self, x, sigma):
#         N, _, H, W = x.size()
#         p = self.patch_size
#         x = self.dct(x)

#         isNonZero = self.udiff(x / (3*sigma))
#         y = x * isNonZero

#         w = 1 / (1+torch.sum(isNonZero, dim=1))
#         w = w.view(N, 1, H+(p-1), W+(p-1))
#         x = nn.functional.conv2d(y, torch.inverse(self.dct.weight.view(p**2, p**2)).view(p**2, p**2, 1, 1))
#         x = w * x
#         x = self.fold(x)
#         x = x[:, :, p-1:H+(p-1), p-1:W+(p-1)]
#         divisor = self.small_conv(w)
#         return x / divisor