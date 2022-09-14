import os
import sys
import torch
import numpy as np

from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix, scalar_mul_matrix

sys.path.append(os.path.dirname(os.getcwd()))
import torch.nn as nn
from utils import euclidean_metric
from networks.convnet import ConvNet
from networks.ResNet import resnet18, resnet34
from networks.DenseNet import densenet121
from networks.WideResNet import wideres
from networks.resnet12 import resnet12
from networks.bigres12 import bigres12

from models.controller import Controller
from models.rerank_Controller import rerank_Controller
from models.support_Controller import support_Controller

import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_name = args.model
        
        if model_name == 'convnet':
            self.encoder = ConvNet(z_dim=args.dim)
        elif model_name == 'resnet18':
            self.encoder = resnet18(remove_linear=True)
        elif model_name == 'resnet34':
            self.encoder = resnet34(remove_linear=True)
        elif model_name == 'densenet121':
            self.encoder = densenet121(remove_linear=True)
        elif model_name == 'wideres':
            self.encoder = wideres(remove_linear=True)
        elif model_name == 'resnet12':
            self.encoder = resnet12()
        elif model_name == 'bigres12':
            self.encoder = bigres12()
                            
        
        self.e2p = ToPoincare(c=1, train_c=args.train_c, train_x=args.train_x)

        self.controller = Controller( args.dim, 128,64,5, args.l, args.divide)
        self.rerank_controller = rerank_Controller(self.args.rerank*2+2, self.args.rerank, self.args.rerank+1)
        self.support_controller = support_Controller(self.args.shot*self.args.shot, self.args.shot, self.args.shot)


        self.proj_k=nn.Linear(args.dim,args.dim)
        self.proj_q=nn.Linear(args.dim,args.dim)
        self.proj_v=nn.Linear(args.dim,args.dim)

        nn.init.normal_(self.proj_k.weight, mean=0, std=np.sqrt(2.0 / (args.dim + args.dim)))
        nn.init.normal_(self.proj_q.weight, mean=0, std=np.sqrt(2.0 / (args.dim + args.dim)))
        nn.init.normal_(self.proj_v.weight, mean=0, std=np.sqrt(2.0 / (args.dim + args.dim)))

        self.layer_norm = nn.LayerNorm(args.dim)
        self.layer_norm2 = nn.LayerNorm(args.dim)

        self.fc_new = nn.Linear(args.dim, args.dim)
        nn.init.xavier_normal_(self.fc_new.weight)

        self.dropout = nn.Dropout(0.5)
        self.dropout_att = nn.Dropout(0.1)
        self.softmax=nn.Softmax()


    def forward(self, shot, query):
        data_shot = self.encoder(shot)
        data_query = self.encoder(query)

        if self.args.setting=='inductive':
            rerank_data=data_shot
        else:
            rerank_data=torch.cat([data_query,data_shot],dim=0)

        rerank_data=rerank_data.repeat(self.args.multihead,1)
        rerank_num=rerank_data.shape[0]


        if self.training:
            data_shot_category = data_shot.reshape(self.args.shot, self.args.way, -1)
        else:
            data_shot_category = data_shot.reshape(self.args.shot, self.args.validation_way, -1)

        mean_proto_category=torch.mean(data_shot_category,0)

        all_data = torch.mean(rerank_data,0)
        all_data = all_data.repeat(mean_proto_category.shape[0],1)


        c=self.controller(  (all_data*(data_shot.shape[0])-mean_proto_category*self.args.shot)/(data_shot.shape[0]-self.args.shot) ,all_data)

        if self.training:
            dis_mat=torch.zeros(self.args.way,rerank_num).cuda()
        else:
            dis_mat=torch.zeros(self.args.validation_way,rerank_num).cuda()

        proto_p=torch.randn(mean_proto_category.shape).cuda()

        if (self.args.model=='resnet12' or self.args.model=='bigres12') and self.args.setting == 'transductive':
            data_shot_prooject_k=mean_proto_category
            data_shot_prooject_q=rerank_data
            data_shot_prooject_v=rerank_data
        if self.args.model=='convnet' and self.args.setting == 'transductive' and self.args.shot==1:
            data_shot_prooject_k=mean_proto_category
            data_shot_prooject_q=rerank_data
            data_shot_prooject_v=rerank_data                   
        else:
            data_shot_prooject_k=self.proj_k(mean_proto_category)
            data_shot_prooject_q=self.proj_q(rerank_data)
            data_shot_prooject_q1=F.relu(data_shot_prooject_q)
            data_shot_prooject_v=self.proj_v(data_shot_prooject_q1)


        for i in range (dis_mat.shape[0]):

            if self.args.shot==1:
                proto_i = self.e2p(data_shot_prooject_k[i].unsqueeze(0),c[i])
                proto_p[i] = proto_i
            else:
                data_shot_i= data_shot_category[:,i,:]
                data_shot_ih = self.e2p(data_shot_i,c[i])
                support_dis_mat=dist_matrix(data_shot_ih,data_shot_ih, c=c[i])
                support_dis_mat=support_dis_mat.view(1,-1)
                support_dis_mat=support_dis_mat / np.power(self.args.dim, 0.5)
                support_dis_mat=self.softmax(-1*support_dis_mat)
                support_weight =self.support_controller(support_dis_mat)
                support_weight_i=(support_weight)*(support_weight.shape[1])
                weight_data_support_i = data_shot_i * (support_weight_i.squeeze().unsqueeze(1))
                weight_data_support_ih = self.e2p(weight_data_support_i,c[i])
                proto_i = poincare_mean(weight_data_support_ih, dim=0, c=c[i]).unsqueeze(0)
                proto_p[i] = proto_i

            support_i = self.e2p(data_shot_prooject_q,c[i])
            dis_mat[i]= dist_matrix(proto_i,support_i, c=c[i])

        #--------------------------------------------------------------------------------------------------------------------------
        sorted, indices = torch.sort(dis_mat)
        test_proto=torch.zeros(proto_p.shape).cuda()
        for i in range(dis_mat.shape[0]):
            #print('-----------------------------')
            n_i_d=dis_mat[i,indices[i,0:self.args.rerank]]
            o_i_d=dis_mat[i,indices[i,self.args.rerank:dis_mat.shape[1]]].mean()
            n_o_d=torch.cat([dis_mat[0:i,indices[i,0:self.args.rerank]], dis_mat[i+1:dis_mat.shape[0],indices[i,0:self.args.rerank]]],dim=0).mean(dim=0)
            o_o_d=torch.cat([dis_mat[0:i,indices[i, self.args.rerank:dis_mat.shape[1] ]], dis_mat[i+1:dis_mat.shape[0],self.args.rerank:dis_mat.shape[1]]],dim=0).mean()
            n_i_d=n_i_d / np.power(self.args.dim, 0.5)
            n_i_d=self.softmax(-1*n_i_d)
            n_o_d=n_o_d / np.power(self.args.dim, 0.5)
            n_o_d=self.softmax(-1*n_o_d)
            i_weight, old_new_weight =self.rerank_controller(torch.cat([n_i_d,n_o_d,o_i_d.unsqueeze(0),o_o_d.unsqueeze(0)],dim=0))
            
            i_weight=(i_weight*old_new_weight[0])*(i_weight.shape[1]+1)
            if (self.args.model=='resnet12' or self.args.model=='bigres12') and self.args.setting == 'transductive':
                weight_data_query_i =rerank_data[indices[i,0:self.args.rerank],:] * (i_weight.squeeze().unsqueeze(1))
            if self.args.model=='convnet' and self.args.setting == 'transductive' and self.args.shot==1:
                weight_data_query_i =rerank_data[indices[i,0:self.args.rerank],:] * (i_weight.squeeze().unsqueeze(1))

            else:
                weight_data_query_i =data_shot_prooject_v[indices[i,0:self.args.rerank],:] * (i_weight.squeeze().unsqueeze(1))
                weight_data_query_i = self.fc_new(weight_data_query_i)
                weight_data_query_i = self.layer_norm(weight_data_query_i)
            weight_data_query_i = self.e2p(weight_data_query_i,c[i])
            mean_i=self.e2p(  (mean_proto_category[i]*(1-old_new_weight)*(i_weight.shape[1]+1)).unsqueeze(0),c[i])
            test_proto[i] = poincare_mean(torch.cat([weight_data_query_i,mean_i],dim=0), dim=0, c=c[i]).unsqueeze(0)
            
        test_proto_norm=test_proto
        data_query_norm=data_query

        new_dis_mat=torch.zeros(dis_mat.shape[0],data_query_norm.shape[0]).cuda()

        for i in range (new_dis_mat.shape[0]):

            query_i= self.e2p(data_query_norm,c[i])
            proto_i=test_proto_norm[i].unsqueeze(0)
            new_dis_mat[i]= dist_matrix(proto_i,query_i, c=c[i]) 

        logits = -new_dis_mat.t() / self.args.temperature

        return logits
