import torch
import torch.nn as nn
#from torch.functional import Tensor
#from torchsummary import summary

"""This script defines the network.
"""

class Network(nn.Module):

    def __init__(self, config):
        super(Network, self).__init__()

        self.layers = nn.ModuleList()
        
        # Input Layer
        self.layers.append(input_layer(config.start_filters))

        # Residual Blocks (HIDDEN)
        for i in range(3):
            in_filters = config.start_filters*(2**i)
            out_filters = in_filters*2
            stride = 1 if i == 0 else 2
            #stride = 2
            self.layers.append(stack_layer(config.block_size, in_filters, out_filters, stride))
        
        # Output Layer
        self.layers.append(output_layer(8, out_filters, config.num_classes))
        
    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs


class conv_block(nn.Module):

    def __init__(self, in_filters, out_filters, stride, kernel_size=3):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=1)
        nn.init.kaiming_normal_(self.conv.weight)
        self.batch_norm = nn.BatchNorm2d(num_features=out_filters, eps=1e-5, momentum=0.997)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.batch_norm(outputs)
        outputs = self.relu(outputs)

        return outputs


class input_layer(nn.Module):

    def __init__(self, filters):
        super(input_layer, self).__init__()

        self.conv = conv_block(3, filters, 1)

        # Bottleneck channels
        self.layers = nn.ModuleList()

        # Reduce channels
        for i in range(2):
            in_filters = filters//(2**i)
            out_filters = in_filters//2
            self.layers.append(conv_block(in_filters, out_filters, 1))

        # Increase channels
        for i in range(2):
            in_filters = out_filters
            out_filters = in_filters*2
            self.layers.append(conv_block(in_filters, out_filters, 1))

    def forward(self, inputs):
        inputs = self.conv(inputs)

        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        outputs = outputs + inputs

        return outputs


class stack_layer(nn.Module):

    def __init__(self, block_size, in_filters, out_filters, strides):
        super(stack_layer, self).__init__()

        self.stack = nn.ModuleList()
        for i in range(block_size):
            if i == 0:
                stride = strides
                projection = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=1, stride=stride, padding=0)
            else:
                stride = 1
                projection = nn.Identity()
            self.stack.append(standard_block(in_filters, out_filters, stride, projection))
            in_filters = out_filters

    def forward(self, inputs):
        outputs = inputs
        for layer in self.stack:
            outputs = layer(outputs)

        return outputs


class standard_block(nn.Module):

    def __init__(self, in_filters, out_filters, stride, projection):
        super(standard_block, self).__init__()

        self.projection = projection
        self.conv1 = conv_block(in_filters, out_filters, stride)
        self.conv2 = conv_block(out_filters, out_filters, 1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = outputs + self.projection(inputs)

        return outputs


class output_layer(nn.Module):

    def __init__(self, size, filters, classes):
        super(output_layer, self).__init__()

        self.pool = nn.AvgPool2d(size)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(filters, classes)
    
    def forward(self, inputs):
        outputs = self.pool(inputs)
        outputs = self.flatten(outputs)
        outputs = self.linear(outputs)

        return outputs

