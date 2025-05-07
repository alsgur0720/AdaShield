import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# Define helper functions similar to `conv2d_bn_lrelu_cf_d` for PyTorch
class DRS(nn.Module):
    def __init__(self, in_channels=21, segt_class=4, n_slice=21, n_filters=[16, 32, 64, 128, 256]):
        super(DRS, self).__init__()
        
        # Encoding path
        self.enc1 = nn.Sequential(
            Conv2dBNLeakyReLU(in_channels, n_filters[0]),
            Conv2dBNLeakyReLU(n_filters[0], n_filters[0])
        )
        
        self.enc2 = nn.Sequential(
            Conv2dBNLeakyReLU(n_filters[0], n_filters[1], stride=2),
            Conv2dBNLeakyReLU(n_filters[1], n_filters[1])
        )
        
        self.enc3 = nn.Sequential(
            Conv2dBNLeakyReLU(n_filters[1], n_filters[2], stride=2),
            Conv2dBNLeakyReLU(n_filters[2], n_filters[2])
        )
        
        self.enc4 = nn.Sequential(
            Conv2dBNLeakyReLU(n_filters[2], n_filters[3], stride=2),
            Conv2dBNLeakyReLU(n_filters[3], n_filters[3])
        )
        
        self.enc5 = nn.Sequential(
            Conv2dBNLeakyReLU(n_filters[3], n_filters[4], stride=2),
            Conv2dBNLeakyReLU(n_filters[4], n_filters[4]),
            Conv2dBNLeakyReLU(n_filters[4], n_filters[2])
        )
        
        self.net1 = Conv2dBNLeakyReLU(n_filters[0], n_filters[1])
        self.net1_rec = Conv2dBNLeakyReLU(n_filters[0], n_filters[1])
        self.net2 = Conv2dBNLeakyReLU(n_filters[1], n_filters[1])
        self.net3 = Conv2dBNLeakyReLU(n_filters[2], n_filters[1])
        self.net4 = Conv2dBNLeakyReLU(n_filters[3], n_filters[1])
        
        
        # self.bottleneck = nn.Sequential(
        #     Conv2dBNLeakyReLU(n_filters[3], n_filters[4], stride=2),
        #     Conv2dBNLeakyReLU(n_filters[4], n_filters[4]),
        #     Conv2dBNLeakyReLU(n_filters[4], n_filters[1])
        # )

        # Decoding_seg path
        self.upconv4 = nn.ConvTranspose2d(n_filters[1], n_filters[1], kernel_size=16, stride=16, bias=False)
        self.upconv3 = nn.ConvTranspose2d(n_filters[1], n_filters[1], kernel_size=8, stride=8, bias=False)
        self.upconv2 = nn.ConvTranspose2d(n_filters[1], n_filters[1], kernel_size=4, stride=4, bias=False)
        self.upconv1 = nn.ConvTranspose2d(n_filters[1], n_filters[1], kernel_size=2, stride=2, bias=False)

        # Decoding_rec path
        self.upconv4_rec = nn.ConvTranspose2d(n_filters[1], n_filters[1], kernel_size=16, stride=16, bias=False)
        self.upconv3_rec = nn.ConvTranspose2d(n_filters[1], n_filters[1], kernel_size=8, stride=8, bias=False)
        self.upconv2_rec = nn.ConvTranspose2d(n_filters[1], n_filters[1], kernel_size=4, stride=4, bias=False)
        self.upconv1_rec = nn.ConvTranspose2d(n_filters[1], n_filters[1], kernel_size=2, stride=2, bias=False)


        # Final output_seg layer
        self.conv_final1 = Conv2dBNLeakyReLU(160, 64)
        self.conv_final2 = Conv2dBNLeakyReLU(64, 64)
        self.output_layer = nn.Conv2d(64, segt_class * n_slice, kernel_size=1)
        
        # Final output_rec layer
        self.conv_final1_rec = Conv2dBNLeakyReLU(160, 64)
        self.conv_final2_rec = Conv2dBNLeakyReLU(64, 64)
        self.output_layer_rec = nn.Conv2d(64, n_slice, kernel_size=1)

    def forward(self, x):
        # Encoding path
        
        x =  torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1)).real
        
        x = torch.abs(x)
        
        
        e1 = self.enc1(x)  # 288x288 -> 288x288, n_filters[0]
        net1_seg = self.net1(e1)
        net1_rec = self.net1_rec(e1)
        e2 = self.enc2(e1) # 288x288 -> 144x144, n_filters[1]
        net2 = self.net2(e2)
        e3 = self.enc3(e2) # 144x144 -> 72x72, n_filters[2]
        net3 = self.net3(e3)
        e4 = self.enc4(e3) # 72x72 -> 36x36, n_filters[3]
        net4 = self.net4(e4)
        b = self.enc5(e4) # 36x36 -> 18x18, n_filters[4]
        
        b_rec = b[:,0:32,:,:].clone()
        b_seg = b[:,32:64,:,:].clone()
        


        
        b_rec = torch.fft.ifft2(torch.fft.ifftshift(b_rec, dim=(-2, -1)), dim=(-2, -1)).real
        b_seg = torch.fft.ifft2(torch.fft.ifftshift(b_seg, dim=(-2, -1)), dim=(-2, -1)).real
      
        B, C, H, W = b_seg.size()    
        
        flattened = b_seg.view(B, C, -1)
        combined = flattened.permute(1, 0, 2).contiguous().view(C, -1)
        mean = combined.mean(dim=1, keepdims=True)

        # Subtract mean
        centered = combined - mean
        cov_matrix = (centered @ centered.T) / (centered.size(1) - 1)
        
        
        #### segt path ####
        b_up = self.upconv4(b_seg)

        net4_seg = self.upconv3(net4)
        net3_seg = self.upconv2(net3)
        net2_seg = self.upconv1(net2)
        
        
        b_up = torch.cat((net1_seg, net2_seg, net3_seg, net4_seg, b_up), dim=1)

        
        # Final convolution layers
        d = self.conv_final1(b_up)
        d = self.conv_final2(d)
        
        # Output layer
        logits_segt = self.output_layer(d)  # Final output layer
        
        logits_segt = logits_segt.permute(0,2,3,1)
        
        logits_segt = logits_segt.contiguous().view(logits_segt.size(0), logits_segt.size(1), logits_segt.size(2), 21, 4)            
        
        #### segt path ####
        
        
        #### rec path ####
        b_up_rec = self.upconv4_rec(b_rec)

        net4_rec = self.upconv3_rec(net4)
        net3_rec = self.upconv2_rec(net3)
        net2_rec = self.upconv1_rec(net2)
        
        
        b_up_rec = torch.cat((net1_rec, net2_rec, net3_rec, net4_rec, b_up_rec), dim=1)

        
        # Final convolution layers
        d_rec = self.conv_final1_rec(b_up_rec)
        d_rec = self.conv_final2_rec(d_rec)
        
        # Output layer
        logits_rec = self.output_layer_rec(d_rec)  # Final output layer
        
        #### rec path ####
        
        
        return logits_segt, cov_matrix, b_seg, logits_rec, b_rec


# Helper class for conv2d -> batchnorm -> leaky_relu sequence
class Conv2dBNLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dBNLeakyReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x
    
class differential_detection(nn.Module):
    def __init__(self, input_size = 1576, hidden_size = 800, output_size= 8):
        super(differential_detection, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.model(x)   
