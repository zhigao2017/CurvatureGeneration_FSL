import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F


class support_Controller(torch.nn.Module):
    
    def __init__(self, input_dim,hidden_dim,output_dim):
        super(support_Controller,self).__init__()

        self.proj1=nn.Linear(input_dim,hidden_dim)
        self.proj2=nn.Linear(hidden_dim,output_dim)
        self.proj3=nn.Linear(output_dim,output_dim)

        nn.init.kaiming_normal_(self.proj1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj3.weight, mode='fan_out', nonlinearity='relu')

        self.softmax=nn.Softmax()

        
    def forward(self,input_dis):


        input=input_dis.view(1,-1)
        input = F.normalize(input)

        output = self.proj1(input)
        output = F.relu(output)
        output = self.proj2(output)
        output = F.relu(output)
        output = self.proj3(output)
        n_weight = self.softmax(output)
        
        return n_weight