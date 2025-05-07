import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class TimeBlock(nn.Module):
    """
    Temporal Gated Convolution on 
    each node of a graph in isolation.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.5, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))


    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2) # (batch_size, num_features, num_nodes, num_timesteps)
        # In analogy to the computer vision,
        # batch_size is the number of images,
        # num_features is the number of channels (usually 3 for RGB images),
        # num_nodes is the height (number of pixels in the image),
        # num_timesteps is the width (number of pixels in the image).
        # So it is like finding the pattern in num_timesteps x num_nodes
        # where the pattern is learned by 2 (=num_features) kernels at each step.
        # Since this kernel is 1D, it finds the pattern only for the time dimention.
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1) # (batch_size, num_nodes, num_timesteps, num_features)
        return out


class STConvBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, dropout_rate=0.5):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STConvBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   dropout_rate=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.dropout2 = nn.Dropout(dropout_rate)
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels,
                                   dropout_rate=dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        t = self.dropout1(t)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t2 = self.dropout2(t2)
        t3 = self.temporal2(t2)
        t3 = self.dropout3(t3)
        return self.batch_norm(t3)
        # return t3

class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).

    Input shape (num_nodes, num_features, num_timestamp_input, num_timestamp_output)
    Output shape (num_nodes, num_timestamp_output)
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, dropout_rate = 0.5):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        '''
        self.block1 = STConvBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STConvBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)
        '''
        self.block1 = STConvBlock(in_channels=num_features, out_channels=16,
                                 spatial_channels=4, num_nodes=num_nodes,
                                 dropout_rate=dropout_rate)
        self.block2 = STConvBlock(in_channels=16, out_channels=16,
                                 spatial_channels=4, num_nodes=num_nodes,
                                 dropout_rate=dropout_rate)
        self.last_temporal = TimeBlock(in_channels=16, out_channels=16,
                                       dropout_rate=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fully = nn.Linear((num_timesteps_input - 2*5) * 16, num_timesteps_output)
        # TODO: change the output shape to (num_timesteps_output, num_features) This makes error now.
        # self.fully = nn.Linear((num_timesteps_input - 2*5) * 16, (num_timesteps_output, num_features))
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.dropout1(out3)
        out5 = self.fully(out4.reshape((out4.shape[0], out4.shape[1], -1)))
        out6 = self.dropout2(out5)
        return out6
