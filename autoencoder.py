import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

class ConvolutionalAutoencoder(torch.nn.Module):

     def __init__(self):
         super(ConvolutionalAutoencoder, self).__init__()
        
        
         ### ENCODER  
         self.conv_1 = torch.nn.Conv2d(reduced_dim,
                                       out_channels=4,   #not sure about this param, can remove
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=0)
                   
        
         ### DECODER                         
         self.deconv_1 = torch.nn.ConvTranspose2d(expand_dim,
                                                  out_channels=4,
                                                  kernel_size=(1, 1),
                                                  stride=(1, 1),
                                                  padding=0)
        
         
     def forward(self, x):
         ### ENCODER
         x = self.conv_1(x)
         x = F.leaky_relu(x)
          
         ### DECODER
         x = self.deconv_1(x)  
         x = F.leaky_relu(x)
         x = x[:, :, :-1, :-1]
         x = torch.sigmoid(x)
         return x