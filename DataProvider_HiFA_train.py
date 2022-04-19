import numpy as np
import os
import glob
from torch.utils.data import Dataset, Sampler
import random

class HiFADataset(Dataset):
    def __init__(self, np_feat_dir = None):
        super().__init__()
        if np_feat_dir is None:
            np_feat_dir = '/imgsemantic_feat/'
        self.np_feat_dir = np_feat_dir
        self.get_feat_list()

    def get_feat_list(self):
        feat_list_dir = list()
        feat_list_dir.append('/00/')
        featname_list = list()
        featname_list.extend(glob.glob(os.path.join(feat_list_dir[0], '*imgfeat*.npy')))

        print('train data raw length = {}'.format(len(featname_list)))

        self.featname_list = featname_list
        random.shuffle( self.featname_list )

    def __len__(self):
        return len(self.featname_list)

    def __getitem__(self, index):
        feat_filename = self.featename_list[index]
        feat = np.load( feat_filename )
        gtlabel = feat_filename.replace('imgfeat','gtlabel')

        return feat, gtlabel
