import torch
import torch.nn as nn
from torch.autograd import Variable as V
from models.MatrixBiMul import MatrixBiMul
import torch.nn.functional as F


class Controller(torch.nn.Module):
    
    def __init__(self, backbone_input_dim,hidden_dim,output_dim,factor_num,l,divide):
        super(Controller,self).__init__()
                
        self.l=l
        self.divide=divide

        self.backbone_input_dim=backbone_input_dim
        self.output_dim=output_dim
        self.hidden_dim=hidden_dim
        self.factor_num=factor_num

        
        self.proto_linear = nn.Linear(backbone_input_dim,hidden_dim*factor_num)
        self.all_linear = nn.Linear(backbone_input_dim,hidden_dim*factor_num)

        self.fclayer=nn.Linear(hidden_dim,output_dim)
        self.predictor = nn.Linear(output_dim,1)
        
        nn.init.xavier_normal(self.proto_linear.weight)
        nn.init.xavier_normal(self.all_linear.weight)


    def forward(self, mean_proto_category, all_data):
        
        proto_data=self.proto_linear(mean_proto_category)
        all_data=self.all_linear(all_data)
        c = torch.mul(proto_data, all_data)
        
        c = torch.sqrt(F.relu(c)) - torch.sqrt(F.relu(-c))  
        c = F.normalize(c, p=2, dim=1) 

        c = c.view(-1, self.hidden_dim, self.factor_num)
        c = torch.squeeze(torch.sum(c, 2))
        c = F.relu(c)
        c = self.fclayer(c)
        c = F.relu(c)

        output9 = self.predictor(c)
        output10=torch.randn(output9.shape).cuda()
        output10=F.sigmoid(output9*self.l)/self.divide

        return output10