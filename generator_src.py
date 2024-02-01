import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#generator: noise~N(0,1) -> Image 0-255

class ScaleLayer(nn.Module):
    
   def __init__(self, scale=255):
       super().__init__()
       self.scale = scale

   def forward(self, input):
       return input * self.scale

class Generator(nn.Module):
    def __init__(self,z_dim,img_dim,conditional):
        super(Generator, self).__init__()
        self.z_dim=z_dim
        self.img_dim=img_dim
        self.conditional=conditional
        feature_dim=2048
        layers=[
            nn.ConvTranspose2d( z_dim,feature_dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2)
        ]
        current_dim=4
        while current_dim<(self.img_dim/2):
            layers.append(nn.ConvTranspose2d( feature_dim,feature_dim//2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(feature_dim//2))
            layers.append(nn.LeakyReLU(0.2))
            current_dim*=2
            feature_dim//=2
        layers.append(nn.ConvTranspose2d( feature_dim,3, 4, 2, 1, bias=False))
        layers.append(nn.Sigmoid())
        #layers.append(ScaleLayer())
        if self.conditional:
            self.conditional_linear=nn.Sequential(
                nn.Linear(768,512),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(512,256),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(256,z_dim),
            )
            self.conditional_mha=nn.MultiheadAttention(z_dim, 4) #query = image kv=text
            self.flatten=nn.Flatten()
            self.unflatten=nn.Unflatten(1,(z_dim,1,1))
        self.main=nn.Sequential(*layers)

    def forward(self,z,text_encoding):
        if self.conditional:
            text_output=self.conditional_linear(text_encoding)
            z=self.flatten(z)
            (z, attn_output_weights)=self.conditional_mha(z, text_output, text_output)
            z=self.unflatten(z)
        return self.main(z)
