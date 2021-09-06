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
#from networks.test_convnet import ConvNet
from networks.ResNet import resnet18, resnet34
from networks.DenseNet import densenet121
from networks.WideResNet import wideres
from networks.resnet12 import resnet12
from networks.newres12 import newres12

from models.controller import Controller
from models.rerank_Controller import rerank_Controller
from models.support_Controller import support_Controller
# +
# global model_dict = {'convnet': ConvNet(z_dim=args.dim), 
#                      'resnet18': resnet18(remove_linear=True),
#                      'densenet121': densenet121(remove_linear=True)}
# -

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
        elif model_name == 'newres12':
            self.encoder = newres12()

        if args.hyperbolic:
            self.e2p = ToPoincare(c=args.c, train_c=args.train_c, train_x=args.train_x)

        self.controller = Controller( args.dim, 128,64,5, args.l, args.divide)
        self.rerank_controller = rerank_Controller(self.args.rerank*2+2, self.args.rerank, self.args.rerank+1)

        self.support_controller = support_Controller(self.args.shot*self.args.shot, self.args.shot, self.args.shot)

        self.softmax=nn.Softmax()
        
    def compute_secondorder(self,inputs_):

        n=inputs_.shape[0]
        p=inputs_.shape[1]
        g=torch.randn(p+1,p+1).cuda()

        m=torch.mean(inputs_,0)
        inputs_wom=inputs_-m
        s=(torch.mm(inputs_wom.t(),inputs_wom))/n
        g[0:p,0:p]=s
        g[p,0:p]=m
        g[0:p,p]=m
        g[p,p]=1

        return g

    def forward(self, data_shot, data_query):

        
        data_shot = self.encoder(data_shot)
        data_query = self.encoder(data_query)

        proto = self.e2p(data_shot,1)
        proto = proto.reshape(self.args.shot, self.args.way, -1)
        proto = poincare_mean(proto, dim=0, c=self.e2p.c)
        data_query = self.e2p((data_query),1)
        logits = -dist_matrix(data_query, proto, c=self.e2p.c) / self.args.temperature

        
        all_data =torch.cat([data_shot,data_query],dim=0)
        all_data = torch.mean(all_data,0)
        
        if self.training:
            data_shot_category = data_shot.reshape(self.args.shot, self.args.way, -1)
        else:
            data_shot_category = data_shot.reshape(self.args.shot, self.args.validation_way, -1)

        mean_proto_category=torch.mean(data_shot_category,0)
        all_data = all_data.repeat(mean_proto_category.shape[0],1)

        c=self.controller(  (all_data*(mean_proto_category.shape[0]+data_query.shape[0])-mean_proto_category*self.args.shot)/(mean_proto_category.shape[0]+data_query.shape[0]-self.args.shot) ,all_data)
        

        #--------------------------------------------------------------------------------------------------------------------------

        if self.training:
            dis_mat=torch.zeros(self.args.way,data_query.shape[0]).cuda()
        else:
            dis_mat=torch.zeros(self.args.validation_way,data_query.shape[0]).cuda()

        proto_p=torch.randn(mean_proto_category.shape).cuda()
        query_p=torch.randn(proto_p.shape[0],data_query.shape[0],data_query.shape[1]).cuda()
        test_proto=torch.randn(mean_proto_category.shape).cuda()
        for i in range (dis_mat.shape[0]):
            if i==0:
                print('c',c[i])

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
            
            
            
            query_i = self.e2p(data_query,c[i])
            query_p[i]=query_i

            dis_mat[i]= dist_matrix(proto_i,query_i, c=c[i])

        #--------------------------------------------------------------------------------------------------------------------------
            sorted, indices = torch.sort(dis_mat)
            

            n_i_d=dis_mat[i,indices[i,0:self.args.rerank]]
            o_i_d=dis_mat[i,indices[i,self.args.rerank:dis_mat.shape[1]]].mean()
            n_o_d=torch.cat([dis_mat[0:i,indices[i,0:self.args.rerank]], dis_mat[i+1:dis_mat.shape[0],indices[i,0:self.args.rerank]]],dim=0).mean(dim=0)
            o_o_d=torch.cat([dis_mat[0:i,indices[i, self.args.rerank:dis_mat.shape[1] ]], dis_mat[i+1:dis_mat.shape[0],self.args.rerank:dis_mat.shape[1]]],dim=0).mean()


            n_i_d=n_i_d / np.power(self.args.dim, 0.5)
            n_i_d=self.softmax(-1*n_i_d)
            n_o_d=n_o_d / np.power(self.args.dim, 0.5)
            n_o_d=self.softmax(-1*n_o_d)


            i_weight, old_new_weight =self.rerank_controller(torch.cat([n_i_d,n_o_d,o_i_d.unsqueeze(0),o_o_d.unsqueeze(0)],dim=0))


            
            i_weight=(i_weight*old_new_weight[0])*(i_weight.shape[1]+self.args.shot)
            weight_data_query_i =data_query[indices[i,0:self.args.rerank],:] * (i_weight.squeeze().unsqueeze(1))
            weight_data_query_i = self.e2p(weight_data_query_i,c[i])

            support_weight_i=(support_weight*(1-old_new_weight[0]))*(support_weight.shape[1]+i_weight.shape[1])
            weight_data_support_i = data_shot_i * (support_weight_i.squeeze().unsqueeze(1))
            weight_data_support_ih = self.e2p(weight_data_support_i,c[i])
            test_proto[i] = poincare_mean(torch.cat([weight_data_query_i,weight_data_support_ih],dim=0), dim=0, c=c[i]).unsqueeze(0)
            


        #--------------------------------------------------------------------------------------------------------------------------
        new_dis_mat=torch.zeros(dis_mat.shape).cuda()
        for i in range (new_dis_mat.shape[0]):
            new_dis_mat[i]= dist_matrix(test_proto[i].unsqueeze(0),query_p[i], c=c[i]) 

        logits = -new_dis_mat.t() / self.args.temperature

        
        return logits
