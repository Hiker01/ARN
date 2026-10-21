# -*- coding: UTF-8 -*-
import torch.utils.data as data
import numpy as np
import torch as t


'''
获取一个视频的LSTM输入数据，和label
'''

class LSTM_DATA(data.Dataset):
    def __init__(self,mode):
        super(LSTM_DATA, self).__init__()
        self.mode = mode
        self.data_label = np.load("/home/hiker/tsn-pytorch/%s" % mode)

    def __getitem__(self, index):
        self.data = self.data_label[index][:][0]
        self.label = self.data_label[index][:][1]
        return t.from_numpy(self.data), self.label

    def __len__(self):
        return len(self.data_label)