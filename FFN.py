"""
Note: Universal Point wise feedforward neural network
Author: Yuhang He
"""
import torch
import torch.nn as nn

class PointWiseFeedForwardNet(nn.Module):
    def __init__(self, in_features=256, out_features=256, dff=1024):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dff = dff

        self.ffn = nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=dff, bias=True, dtype=torch.float32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=dff, out_features=out_features, bias=True, dtype=torch.float32)
        )

    def forward(self, x):
        '''
        the input x shape is [B,N,obs_num,feat_len]
        :param x: input feature after multi-head self-attention module
        :return: output features
        '''
        feat = self.ffn(x)  # [B,N,obs_num, feat_len]

        return feat