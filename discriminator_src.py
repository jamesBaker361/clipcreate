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

class Discriminator(nn.Module):
    def __init__(self, image_dim,init_dim,final_dim,style_list):
        super(Discriminator, self).__init__()
        self.image_dim=image_dim
        self.init_dim=init_dim
        self.final_dim=final_dim
        layers=[
            nn.Conv2d(3, init_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        while init_dim<final_dim:
            layers.append(nn.Conv2d(init_dim, init_dim * 2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(init_dim*2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            init_dim*=2
            image_dim//=2
        layers.append(nn.Flatten())
        self.main = nn.Sequential(
            * layers
        )
        self.binary_layers=nn.Sequential(
            nn.Linear(init_dim*image_dim*image_dim//4,1),
            nn.Sigmoid()
        )
        self.style_layers=nn.Sequential(
            nn.Linear(init_dim*image_dim*image_dim//4,1024),
            nn.Linear(1024,512),
            nn.Linear(512, len(style_list))
        )

    def forward(self, input):
        main_output = self.main(input)
        return self.binary_layers(main_output), self.style_layers(main_output)