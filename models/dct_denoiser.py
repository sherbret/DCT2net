#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : DCT2Net
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2021, v1.0.

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def C(x):
    if x == 0:
        return 1 / np.sqrt(2)
    return 1.0

class DCT_denoiser(nn.Module):
    """ Original DCT denoiser with adaptative aggregation as in BM3D """
    def __init__(self, patch_size=16):
        super(DCT_denoiser, self).__init__()

        self.patch_size = patch_size
        p = patch_size
        P = np.zeros((p**2, p**2))
        for x in range(p):
            for y in range(p):
                for i in range(p):
                    for j in range(p):
                        P[x*p + y, i*p + j] +=  2 / p * C(i) * C(j) * np.cos((2*x + 1)*i*np.pi/(2*p)) * np.cos((2*y + 1)*j*np.pi/(2*p))

        P = P.astype(np.float32)
        self.P = nn.Parameter(torch.from_numpy(P))
        self.P.requires_grad_(False)
        
        Pm1 = P.T 
        self.Pm1 = nn.Parameter(torch.from_numpy(Pm1))
        self.Pm1.requires_grad_(False)

        self.threshold = nn.Hardshrink(1.0)
        self.unfold = nn.Unfold(kernel_size=patch_size)


    def forward(self, x, sigma):
          N, _, H, W = x.size()
          p = self.patch_size
          
          x = self.unfold(x)
          x = self.Pm1 @ x
          
          x = x / (3*sigma)
          x = self.threshold(x)
          x = x * (3*sigma)
          
          count_nz = torch.count_nonzero(x[:, 1:, :], dim=1)
          w = 1/(1+count_nz)
          x = self.P @ x

          x *= w 
          x = torch.nn.functional.fold(x, output_size=(H, W), kernel_size=p)
          
          ones_patches = torch.ones(N, p**2, (H-p+1) * (W-p+1), device=device)
          divisor = torch.nn.functional.fold(w * ones_patches, output_size=(H, W), kernel_size=p)
          return x / divisor

