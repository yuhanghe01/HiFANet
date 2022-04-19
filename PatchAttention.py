'''
Note: Patch Attention Merges kxk featue map into 1 feature
Author: Yuhang He
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import FFN
import numpy as np


class PatchAttentionNet(nn.Module):
    def __init__(self, in_features = 256, point_num = 10, obs_num = 5, KQ_len = 64, num_heads = 4, dropout_ratio = 0.01 ):
        super().__init__()

        self.in_features = in_features
        self.KQ_len = KQ_len
        self.num_heads = num_heads
        self.point_num = point_num
        self.obs_num = obs_num
        self.dropout_ratio = dropout_ratio

        self.K_linear = torch.nn.Linear( in_features = in_features,
                                         out_features = self.KQ_len*self.num_heads,
                                         bias = True,
                                         dtype = torch.float32 )
        self.Q_linear = torch.nn.Linear( in_features = in_features,
                                         out_features = self.KQ_len*self.num_heads,
                                         bias = True,
                                         dtype = torch.float32 )
        self.V_linear = torch.nn.Linear( in_features = self.in_features,
                                         out_features = self.KQ_len*self.num_heads,
                                         bias = True,
                                         dtype = torch.float32 )
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
        :param K_val: pre-computed Key val, [B, N, obs_num, numheads, k, k, KQ_len]
        :param Q_val: pre-computed Val val, [B, N, obs_num, numheads, k, k, KQ_len]
        :return: softmaxed and scaled weight-matrix, [B, N, obs_num, numheads, k, k]
        '''
        patch_len = Q_val.size()[4]
        principle_Q = Q_val[:,:,:,:,patch_len//2:patch_len//2+1, patch_len//2:patch_len//2+1,:]

        # principle_Q_dup = torch.tile( input = principle_Q,
        #                               reps = (1, 1, 1, 1, patch_len, patch_len, 1) )

        principle_Q_dup = torch.tile(principle_Q,
                                     (1, 1, 1, 1, patch_len, patch_len, 1))

        QK_mul = torch.mul( input=principle_Q_dup, other=K_val ) #[B,N,obs_num, numheads,k,k,KQ_len]
        QK_mul = torch.sum(QK_mul, dim=6, keepdim=False)
        QK_mul = torch.div(QK_mul, np.sqrt(float(self.KQ_len))) #[B,N,obs_num, numheads, k,k]

        QK_mul = torch.flatten(input=QK_mul, start_dim=4, end_dim=5) #[B,N,obs_num, numheads,k^2]

        QK_mul = F.softmax( QK_mul, dim=4 )

        QK_mul = torch.reshape( QK_mul, shape=(QK_mul.size()[0],
                                               QK_mul.size()[1],
                                               QK_mul.size()[2],
                                               self.num_heads,
                                               patch_len,
                                               patch_len)) #[B,N,obs_num,k,k]

        return QK_mul

    def merge_patch_feat(self, V, QK_weight ):
        '''
        merge the patch-feature into one single feature according to the pre-computed QK_weight matrix
        :param V: pre-computed value, of shape [B, N, obs_num, numheads, k, k, feat_len], float32
        :param QK_weight: pre-computed weight, of shape [B, N, obs_num, numheads, k, k]
        :return: [B, N, obs_num, numheads, feat_len]
        '''
        QK_weight = torch.unsqueeze( QK_weight, dim = -1 )
        QK_weight = torch.flatten( QK_weight, start_dim = 4, end_dim=5 ) #[B,N, obs_num,numheads, k^2,1]

        V_flattened = torch.flatten( V, start_dim=4, end_dim=5) #[B, N, obs_num, numheads, k^2, feat_len]
        V_trans = V_flattened.permute(0,1,2,3,5,4) ##[B,N,obs_num, numheads,feat_len,k^2]

        merged_feat = torch.matmul( V_trans, QK_weight ) #[B, N, obs_num, numheads, feat_len, 1]

        merged_feat = torch.squeeze( merged_feat, dim = -1 ) #[B, N, obs_num, numheads, feat_len]

        return merged_feat

    def split_to_multiheads(self, inputs ):
        '''
        split the inputs to multiheads
        :param inputs: input tensor to split, [B, N, obs_num, k, k, feat_len]
        :return: splitted tensor, [B, N, obs_num, headnum, k, k, feat_len/head_num]
        '''
        input_shape = inputs.size()
        output_feat = torch.reshape(inputs, shape=(input_shape[0],
                                                   input_shape[1],
                                                   input_shape[2],
                                                   input_shape[3],
                                                   input_shape[4],
                                                   self.num_heads,
                                                   input_shape[-1]//self.num_heads))

        output_feat = output_feat.permute(0,1,2,5,3,4,6)

        return output_feat


    def forward(self, x):
        '''
        Forward computation of the patch attention
        :param x: input tensor, [B, N, obs_num, k, k, feat_len]
        :return: output feature after patch attention, [B, N, feat_len]
        '''
        V_tmp = self.V_linear( x )
        K_tmp = self.K_linear( x )
        Q_tmp = self.Q_linear( x )

        #reshape to multi-heads
        V_tmp = self.split_to_multiheads( V_tmp )
        K_tmp = self.split_to_multiheads( K_tmp )
        Q_tmp = self.split_to_multiheads( Q_tmp )

        QK_mul = self.compute_query_key_mul( K_val=K_tmp, Q_val=Q_tmp )

        merged_feat = self.merge_patch_feat( V=V_tmp, QK_weight=QK_mul ) #[B, N, obs_num, numheads, feat_len]

        #concatenate multiheads feature for form one large feature
        merged_feat = merged_feat.view(merged_feat.size()[0],
                                       merged_feat.size()[1],
                                       merged_feat.size()[2],
                                       merged_feat.size()[3]*merged_feat.size()[4])

        merged_feat_linear = self.feat_linear( merged_feat )

        attent_output = self.dropout(merged_feat_linear)
        output1 = self.layer_norm1( attent_output )

        ffn_output = self.ffn( output1 )
        ffn_output = self.dropout(ffn_output)

        output2 = self.layer_norm2(output1 + ffn_output )

        return output2