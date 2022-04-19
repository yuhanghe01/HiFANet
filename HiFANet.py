'''
Note: Implement HiFANet
Author: Yuhang He
Email: yuhanghe01@gmail.com
'''

import torch
import torch.nn as nn
import PatchAttention
import InstanceAttention
import InterPointAttention

class HiFANet(nn.Module):
    def __init__(self, in_features=256, point_num=10, obs_num=5, KQ_len=64, num_heads=4, dropout_ratio=0.01, class_num=14):
        super().__init__()

        self.in_features = in_features
        self.KQ_len = KQ_len
        self.num_heads = num_heads
        self.point_num = point_num
        self.obs_num = obs_num
        self.dropout_ratio = dropout_ratio
        self.class_num = class_num

        self.patch_attention_net = PatchAttention.PatchAttentionNet( in_features = self.in_features,
                                                      point_num = self.point_num,
                                                      obs_num = self.obs_num,
                                                      KQ_len = self.KQ_len,
                                                      num_heads = self.num_heads,
                                                      dropout_ratio = self.dropout_ratio )

        self.instance_attention_net = InstanceAttention.InstanceAttentionNet( in_features=self.in_features,
                                                            point_num=self.point_num,
                                                            obs_num = self.obs_num,
                                                            KQ_len=self.KQ_len,
                                                            num_heads=self.num_heads,
                                                            dropout_ratio=self.dropout_ratio)

        self.interpoint_attention_net = InterPointAttention.InterPointAttentionNet( in_features=self.in_features,
                                                                point_num=self.point_num,
                                                                obs_num = self.obs_num,
                                                                KQ_len=self.KQ_len,
                                                                num_heads=self.num_heads,
                                                                dropout_ratio=self.dropout_ratio)

        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=512, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=512, out_features=self.class_num, bias=True)
        )

    def forward(self, img_feat, point_loc):
        '''
        HFANet accepts pre-extracted feature from images, and point 3D Location
        :param img_feat: [B, N, obs_num, patch_size, patch_size, feat_len]
        :param point_loc: [B, N, N, 3]
        :param gt_label: [B, N]
        :return: classification loss
        '''
        output_patchattent = self.patch_attention_net( img_feat )
        output_instanceattent = self.instance_attention_net( output_patchattent ) #[B, 1, 256]
        output_interpoint_attent = self.interpoint_attention_net( output_instanceattent, point_loc )

        predict_logits = self.classification_head( output_interpoint_attent )

        return predict_logits