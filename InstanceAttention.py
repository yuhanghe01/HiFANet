'''
Note: Instance Attention Merges kxk featue map into 1 feature
Author: Yuhang He
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import FFN
import numpy as np


class InstanceAttentionNet(nn.Module):
    def __init__(self, in_features=256, point_num=10, obs_num=5, KQ_len=64, num_heads=4, dropout_ratio=0.01):
        super().__init__()

        self.in_features = in_features
        self.KQ_len = KQ_len
        self.num_heads = num_heads
        self.point_num = point_num
        self.dropout_ratio = dropout_ratio
        self.obs_num = obs_num

        self.K_linear = torch.nn.Linear(in_features=in_features,
                                        out_features=self.KQ_len * self.num_heads,
                                        bias=True,
                                        dtype=torch.float32)

        self.Q_linear = torch.nn.Linear(in_features=in_features,
                                        out_features=self.KQ_len * self.num_heads,
                                        bias=True,
                                        dtype=torch.float32)

        self.V_linear = torch.nn.Linear(in_features=self.in_features,
                                        out_features=self.KQ_len * self.num_heads,
                                        bias=True,
                                        dtype=torch.float32)

        self.feat_linear = torch.nn.Linear(in_features=self.in_features,
                                           out_features=self.in_features,
                                           bias=True,
                                           dtype=torch.float32)

        self.layer_norm1 = torch.nn.LayerNorm(normalized_shape=[self.point_num, self.obs_num, self.in_features])
        self.layer_norm2 = torch.nn.LayerNorm(normalized_shape=[self.point_num, self.obs_num, self.in_features])
        self.dropout = torch.nn.Dropout(p=self.dropout_ratio)

        self.ffn = FFN.PointWiseFeedForwardNet(in_features=self.in_features,
                                               out_features=self.in_features,
                                               dff=1024)

    def compute_query_key_mul(self, K_val, Q_val):
        '''
        compute the principle Q_val's dot product similarity w.r.t. all K_val
        :param K_val: pre-computed Key val, [B, N, obs_num, numheads, KQ_len]
        :param Q_val: pre-computed Val val, [B, N, obs_num, numheads, KQ_len]
        :return: softmaxed and scaled weight-matrix, [B, N, obs_num, numheads]
        '''
        K_val = K_val.permute(0,1,3,2,4) #[B,N,numheads,obs_num,KQ_len]
        Q_val = Q_val.permute(0,1,3,4,2) #[B,N,numheads,KQ_len,obs_num]

        QK_mul = torch.matmul(input=K_val, other=Q_val)  # [B,N,numheads,obsnum,obsnum]
        QK_mul = torch.div(QK_mul, np.sqrt(float(self.KQ_len)))  # [B,N,numheads,obs_num,obs_num]
        QK_mul = F.softmax(QK_mul, dim=4)

        return QK_mul

    def merge_instance_feat(self, V, QK_weight):
        '''
        merge the instance-feature into one single feature according to the pre-computed QK_weight matrix
        :param V: pre-computed value, of shape [B, N, obs_num, numheads, feat_len], float32
        :param QK_weight: pre-computed weight, of shape [B, N, numheads, obs_num, obs_num]
        :return: [B, numheads, feat_len]
        '''
        QK_weight = QK_weight.permute(0,1,2,4,3) #guarantee the softmaxed axis is in the second-last dim

        V_permute = V.permute(0,1,3,4,2) #[B, N, numheads, feat_len, obs_num]

        merged_feat = torch.matmul(V_permute, QK_weight)  # [B, N, numheads, feat_len, obsnum]

        return merged_feat

    def split_to_multiheads(self, inputs):
        '''
        split the inputs to multiheads
        :param inputs: input tensor to split, [B, N, obs_num, feat_len]
        :return: splitted tensor, [B, N, obs_num, headnum, feat_len/head_num]
        '''
        input_shape = inputs.size()
        output_feat = inputs.view(input_shape[0],
                                  input_shape[1],
                                  input_shape[2],
                                  self.num_heads,
                                  input_shape[3]//self.num_heads)

        return output_feat

    def forward(self, x):
        '''
        Forward computation of the patch attention
        :param x: input tensor, [B, N, obs_num, feat_len]
        :return: output feature after patch attention, [B, N, feat_len]
        '''
        V_tmp = self.V_linear(x)
        K_tmp = self.K_linear(x)
        Q_tmp = self.Q_linear(x)

        # reshape to multi-heads
        V_tmp = self.split_to_multiheads(V_tmp)
        K_tmp = self.split_to_multiheads(K_tmp)
        Q_tmp = self.split_to_multiheads(Q_tmp)

        QK_mul = self.compute_query_key_mul(K_val=K_tmp, Q_val=Q_tmp)

        merged_feat = self.merge_instance_feat(V=V_tmp, QK_weight=QK_mul) #[B, N, numheads, feat_len, obs_num]
        merged_feat = merged_feat.permute(0,1,4,2,3)

        merged_feat = merged_feat.view(merged_feat.size()[0],
                                       merged_feat.size()[1],
                                       merged_feat.size()[2],
                                       merged_feat.size()[3] * merged_feat.size()[4])

        merged_feat_linear = self.feat_linear(merged_feat)
        attent_output = self.dropout(merged_feat_linear)
        output1 = self.layer_norm1(attent_output + x)

        ffn_output = self.ffn(output1)
        ffn_output = self.dropout(ffn_output)


        output2 = self.layer_norm2(output1 + ffn_output)

        #apply mean average pooling to merge feature across feature observations
        output2 = torch.mean(output2,dim=2,keepdim=False)

        return output2