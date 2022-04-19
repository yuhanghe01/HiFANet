'''
Note: Instance Attention Merges kxk featue map into 1 feature
Author: Yuhang He
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import FFN
import numpy as np

class InterPointAttentionNet(nn.Module):
    def __init__(self, in_features=256, point_num=10, obs_num = 5, KQ_len=64, num_heads=4, dropout_ratio=0.01):
        super().__init__()

        self.in_features = in_features
        self.KQ_len = KQ_len
        self.num_heads = num_heads
        self.point_num = point_num
        self.obs_num = obs_num
        self.dropout_ratio = dropout_ratio

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

        self.layer_norm1 = torch.nn.LayerNorm(normalized_shape=[self.point_num, self.in_features])
        self.layer_norm2 = torch.nn.LayerNorm(normalized_shape=[self.point_num, self.in_features])
        self.dropout = torch.nn.Dropout(p=self.dropout_ratio)

        self.ffn = FFN.PointWiseFeedForwardNet(in_features=self.in_features,
                                               out_features=self.in_features,
                                               dff=1024)

        self.construct_3D_position_encoder()

    def construct_3D_position_encoder(self):
        '''
        pre-construct 3D point cloud relative position encoding, we follow Point Transformer
        (https://arxiv.org/pdf/2012.09164.pdf) to encode inter-point position, which consists of FC-ReLU-FC
        :return: built-in point 3D position encoder
        '''
        self.interpoint_posencoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=128,bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=128, out_features=self.KQ_len*self.num_heads, bias=True)
        )

    def compute_query_key_mul(self, K_val, Q_val, pos_prior):
        '''
        compute the principle Q_val's dot product similarity w.r.t. all K_val
        :param K_val: pre-computed Key val, [B, N, numheads, KQ_len]
        :param Q_val: pre-computed Val val, [B, N, numheads, KQ_len]
        :param pos_prior: pre-computed position prior, [B, N, N, numheads, KQ_len]
        :return: softmaxed and scaled weight-matrix, [B, numheads, N, N]
        '''
        K_val = K_val.permute(0, 2, 1, 3)  # [B,numheads,N,KQ_len]
        Q_val = Q_val.permute(0, 2, 3, 1)  # [B,numheads,KQ_len,N]

        QK_mul = torch.einsum('ijmn,ijnk->ijmkn',K_val, Q_val) #[B, numheads, N, N, KQ_len]
        pos_prior = pos_prior.permute(0,3,1,2,4)
        QK_mul = QK_mul + pos_prior
        QK_mul = torch.sum( QK_mul, dim=4, keepdim=False, dtype=torch.float32 )
        QK_mul = torch.div(QK_mul, np.sqrt(float(self.KQ_len)))  # [B,numheads,N,N]

        QK_mul = F.softmax(QK_mul, dim=3)

        return QK_mul

    def merge_interpoint_feat(self, V, QK_weight):
        '''
        merge the instance-feature into one single feature according to the pre-computed QK_weight matrix
        :param V: pre-computed value, of shape [B, N, numheads, feat_len], float32
        :param QK_weight: pre-computed weight, of shape [B, numheads, N, N]
        :return: [B, numheads, feat_len]
        '''
        QK_weight = QK_weight.permute(0, 1, 3, 2)  # guarantee the softmaxed axis is in the second-last dim

        V_permute = V.permute(0, 2, 3, 1)  # [B, numheads, feat_len, N]

        merged_feat = torch.matmul(V_permute, QK_weight)  # [B, numheads, feat_len, N]

        return merged_feat

    def split_to_multiheads(self, inputs):
        '''
        split the inputs to multiheads
        :param inputs: input tensor to split, [B, N, feat_len]
        :return: splitted tensor, [B, N, headnum, feat_len/head_num]
        '''
        input_shape = inputs.size()
        output_feat = inputs.view(input_shape[0],
                                  input_shape[1],
                                  self.num_heads,
                                  input_shape[2] // self.num_heads)

        return output_feat

    def split_posprior_to_multiheads(self, inputs):
        '''
        split the position prior into multiheads
        :param inputs: pos prior input, of shape [B,N,N,feat_len]
        :return: splited pos prior, of shape [B, N, N, numheads, featlen//numheads]
        '''
        input_shape = inputs.size()
        output_feat = inputs.view(input_shape[0],
                                  input_shape[1],
                                  input_shape[2],
                                  self.num_heads,
                                  input_shape[3]//self.num_heads)

        return output_feat

    def forward(self, input_feat, input_3Dloc):
        '''
        Forward computation of interpoint attention to get the final feature representation for
        :param input_feat: the pre-computed feature, [B, N, feat_len], which is computed by patch attention
            and instance attention module.
        :param input_3Dloc: the relative 3D coordinate difference [B, N, N, 3]
        :return:
        '''
        V_tmp = self.V_linear(input_feat)
        K_tmp = self.K_linear(input_feat)
        Q_tmp = self.Q_linear(input_feat)

        # reshape to multi-heads
        V_tmp = self.split_to_multiheads(V_tmp)
        K_tmp = self.split_to_multiheads(K_tmp)
        Q_tmp = self.split_to_multiheads(Q_tmp)

        pos_prior = self.interpoint_posencoder( input_3Dloc ) #[B, N, N, feat_len]
        pos_prior = self.split_posprior_to_multiheads(pos_prior)

        QK_mul = self.compute_query_key_mul(K_val=K_tmp, Q_val=Q_tmp, pos_prior=pos_prior)

        merged_feat = self.merge_interpoint_feat(V=V_tmp, QK_weight=QK_mul) # [B, numheads, feat_len, N]
        merged_feat = merged_feat.permute(0, 3, 1, 2)

        merged_feat = merged_feat.view(merged_feat.size()[0],
                                       merged_feat.size()[1],
                                       merged_feat.size()[2] * merged_feat.size()[3])

        merged_feat_linear = self.feat_linear(merged_feat)
        attent_output = self.dropout(merged_feat_linear)
        output1 = self.layer_norm1(attent_output + input_feat)

        ffn_output = self.ffn(output1)
        ffn_output = self.dropout(ffn_output)

        output2 = self.layer_norm2(output1 + ffn_output) #[B, N, featlen]

        return output2