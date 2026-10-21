# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
class LSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, out_size=101, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.relu = nn.ReLU()
        #self.linear = nn.Linear(1024, 101)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(7, momentum=0.1)
        #self.out = nn.Linear(hidden_size, out_size)
        #self.drop = nn.Dropout(p=0.8)
        #self.drop = nn.Dropout(p=0.9)

    def forward(self, vet_inputs, hidden = None):
        #inputs = vet_inputs.view(self.batch_size, 25, self.input_size)
        #print(vet_inputs.size())   (batch,25,1024)
        hidden = self.init_hidden(len(vet_inputs))
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])
        relu_out = self.relu(vet_inputs)
        #drop_out = self.drop(vet_inputs)
        #print(drop_out.view(-1,1024).size())     
        #linear_out = self.linear(drop_out.view(-1,1024)).view(-1,25,512)
        #linear_out = self.linear(vet_inputs.view(-1,1024)).view(-1,25,101)
        #print(linear_out.size())
        output, _ = self.lstm(relu_out, hidden)
        output = output .contiguous()
        bn = self.bn(output)
        #drop_out = self.drop(output)
        #output_space = self.out(output)
        # 做出预测
        tag_space = bn[:, -1, :]   #使用最后一个序列的结果
        #tag_space = torch.max(output_space,1)[0] #使用最大融合方式
        #tag_space = torch.mean(output_space,1)  #平均融合序列
        #tags = F.log_softmax(tag_space,dim=1)   #获得softmax的结果
    
        return tag_space
    
    def init_hidden(self, batch_size):
        hidden = (Variable(torch.zeros(batch_size, self.num_layers, self.hidden_size).zero_().cuda()), Variable(torch.zeros(batch_size, self.num_layers, self.hidden_size).zero_().cuda()))
        return hidden