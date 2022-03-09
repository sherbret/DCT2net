#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : DCT2Net
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2021, v1.0.

import torch
import torch.nn as nn
import numpy as np

def C(x):
    if x == 0:
        return 1 / np.sqrt(2)
    return 1.0

class myUDiff(torch.autograd.Function):
    """ Implementation of the function x^(2m) / (x^(2m) + 1) for m = 32. Note that for |x| higer than 1.3, the best approximation is 1.0 """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()

        b = 1.3
        n = 64
        abs_input = torch.abs(input)

        output[abs_input >= b] = 1

        ind = abs_input < b
        x = output[ind]
        y = torch.pow(x, n)
        output[ind] = y / (y + 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        b = 1.3
        n = 64
        abs_input = torch.abs(input)

        grad_input[abs_input >= b] = 0

        ind = abs_input < b
        x = grad_input[ind]
        y = input[ind]
        z = torch.pow(y, n-1)
        grad_input[ind] = x * (n * z / (y*z + 1)**2)

        return grad_input



class DCT2net(nn.Module):
    def __init__(self, patch_size=13):
        super(DCT2net, self).__init__()

        self.patch_size = patch_size
        p = patch_size

        # Initialization of the matrix P
        P = np.zeros((p**2, p**2))
        for x in range(p):
            for y in range(p):
                for i in range(p):
                    for j in range(p):
                        P[x*p + y, i*p + j] +=  2 / p * C(i) * C(j) * np.cos((2*x + 1)*i*np.pi/(2*p)) * np.cos((2*y + 1)*j*np.pi/(2*p))
        P = P.astype(np.float32)
        Pm1 = P.T


        my_filter = torch.zeros(p**2, 1, p, p)
        for k in range(p**2):
            i, j = k // p, k % p
            my_filter[k, 0, i, j] = 1.0

        self.fold = nn.ConvTranspose2d(p**2, 1, p, bias=False)
        self.fold.weight = nn.Parameter(my_filter)
        self.fold.requires_grad_(False)

        self.small_conv = nn.Conv2d(1, 1, p, bias=False)
        self.small_conv.weight = nn.Parameter(torch.ones(1, 1, p, p))
        self.small_conv.requires_grad_(False)

        Pm1_torch = torch.from_numpy(Pm1).view(p**2, 1, p, p).contiguous()
        self.dct = nn.Conv2d(1, p**2, p, bias=False)#, padding=N-1, padding_mode='reflect')
        self.dct.weight = nn.Parameter(Pm1_torch)

        self.udiff = myUDiff.apply



    def forward(self, x, sigma):
        N, _, H, W = x.size()
        p = self.patch_size
        x = self.dct(x)

        isNonZero = self.udiff(x / (3*sigma))
        y = x * isNonZero

        w = 1 / (1+torch.sum(isNonZero, dim=1))
        w = w.view(N, 1, H-(p-1), W-(p-1))
        x = nn.functional.conv2d(y, torch.inverse(self.dct.weight.view(p**2, p**2)).view(p**2, p**2, 1, 1))
        x = w * x
        x = self.fold(x)
        x = x[:, :, p-1:H-(p-1), p-1:W-(p-1)]
        divisor = self.small_conv(w)
        return x / divisor