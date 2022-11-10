# Created on 8/15/22 at 3:14 PM

# Author: Jenny Sun

'''this scrip contains models that fit drift and boundary, single boundary
    model split from the the beggining'''

# Created on 10/12/21 at 10:50 PM

# Author: Jenny Sun
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
from layers_sinc_spatial import *




# torch.manual_seed(2022)
# np.random.seed(2022)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class peakValue(nn.Module):
    '''predicts drift and boundary
    attention layer after sinc and after conv
    prediction splits before sinc layer'''

    # filter_length = 251
    num_filters = 32
    filter_length = 131
    t_length = 500
    pool_window_ms =250   # set the pool window in time unit
    stride_window_ms = 100   # set the stride window in time unit
    pool_window = int(np.rint(((t_length- filter_length +1) * pool_window_ms)/ (1000)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * stride_window_ms)/ (1000)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98
    spatialConvDepth = 1
    attentionLatent = 6
    sr = 500
    cutoff = sr/2  # can set it to nyquist
    def __init__(self, dropout):
        super(peakValue, self).__init__()
        self.b0_value = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d_value= sinc_conv(self.num_filters, self.filter_length, self.sr,  cutoff=self.cutoff)

        self.b_value = nn.BatchNorm2d(self.num_filters, momentum=0.99)

        self.separable_conv_value = SeparableConv2d(self.num_filters, self.num_filters, depth = self.spatialConvDepth, kernel_size= (self.num_chan,1))

        # self.separable_conv_point = SeparableConv2d_pointwise(32*3, 32*3, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        self.b2_value = nn.BatchNorm2d(self.num_filters*self.spatialConvDepth, momentum=0.99)

        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_value = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1_value = torch.nn.Dropout(p=dropout)


        self.fc_value = torch.nn.Linear(self.num_filters*self.spatialConvDepth*self.output_size,1)

        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None


    # attention layers



        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.gap0_value = torch.nn.AdaptiveAvgPool2d(1)
        self.mlp0_value = torch.nn.Sequential(
            torch.nn.Linear(self.num_filters, self.num_filters // self.attentionLatent, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.num_filters // self.attentionLatent, self.num_filters, bias=False),
            torch.nn.Sigmoid()
        )


    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length)
        x = self.b0_value(x)
        x = torch.squeeze(x)
        if batch_n > 1:
            x = torch.squeeze(x)
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x0_value = self.sinc_cnn2d_value(x)


        # start attention for choice
        b_value, c_value, _, _ = x0_value.size()
        y0_value = self.gap0_value(x0_value).view(b_value, c_value)
        y0_value = self.mlp0_value(y0_value).view(b_value, c_value, 1, 1)
        score_new_value = x0_value * y0_value.expand_as(x0_value)
        # end attention

        # spatial convulation layer for choice
        score0_value_ = self.b_value(score_new_value)
        score0_value= self.separable_conv_value(score0_value_) # output is [n, 64,1,1870)


        # relu  layers
        score_value = self.b2_value(score0_value)
        score_value = F.relu(score_value)

        score_value = self.pool1_value(score_value)
        score_value = self.dropout1_value(score_value)  # output is [n, 64,1,,37)


        # output layer to choice
        value = score_value.view(-1,self.num_filters*self.spatialConvDepth*self.output_size)
        value = self.fc_value(value)   # choice
        value = F.relu(value)

        return value

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp





