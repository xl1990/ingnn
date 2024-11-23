import math 

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree,to_undirected

class DiffusionLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

    def forward(self, x, edge_index, norm=None):
        if norm is None:
            edge_index,_ = add_self_loops(edge_index)
#             edge_index = to_undirected(edge_index)
            row, col = edge_index
#             deg = degree(row, x.size(0), dtype=x.dtype)
            deg = degree(col, x.size(0), dtype=x.dtype)

#             print(deg)
            deg_inv_sqrt = deg.pow(-0.5)
            try:
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            except:
                print(deg_inv_sqrt)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    
# normalization methods for tensors of size N*C
    
class BatchNormNC(nn.Module):
    def __init__(self, num_features):
        super(BatchNormNC, self).__init__()
        self.bn = torch.nn.BatchNorm1d(num_features)
        
    def forward(self,x):
        return self.bn(x)
        
class LayerNormNC(nn.Module):
    def __init__(self, num_features):
        super(LayerNormNC, self).__init__()
        self.ln = torch.nn.LayerNorm((1,num_features))
        
    def forward(self,x):
        return self.ln(x.unsqueeze(1)).squeeze() 
    
class InstanceNormNC(nn.Module):
    def __init__(self, num_features):
        super(InstanceNormNC, self).__init__()
        self.sn = nn.InstanceNorm1d(1)
        
    def forward(self,x):
        return self.sn(x.unsqueeze(1)).squeeze()
    
class RescalingNormNC(nn.Module):
    def __init__(self, num_features):
        super(RescalingNormNC, self).__init__()
        self.eps = 1e-12
        
    def forward(self,x):
        numerator = (x-x.min(dim=-1,keepdim=True)[0])
        denominator = (x.max(dim=-1,keepdim=True)[0]-x.min(dim=-1,keepdim=True)[0])
        return numerator/(denominator+self.eps)
    
class MeanNormNC(nn.Module):
    def __init__(self, num_features):
        super(MeanNormNC, self).__init__()
        self.eps = 1e-12
        
    def forward(self,x):
        numerator = (x-x.mean(dim=-1,keepdim=True))
        denominator = (x.max(dim=-1,keepdim=True)[0]-x.min(dim=-1,keepdim=True)[0])
        return numerator/(denominator+self.eps)
    
class PNormNC(nn.Module):
    def __init__(self, num_features, p=2.0):
        super(PNormNC, self).__init__()
        self.p = p
        
    def forward(self,x):
        return F.normalize(x,p=self.p,dim=-1)
    
# -------------------------------------------------------------------
# layers for AdaGNN
# -------------------------------------------------------------------
    
class Adagnn_without_weight(nn.Module):

    def __init__(self, diag_dimension, in_features, out_features, bias=True):
        super(Adagnn_without_weight, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diag_dimension = diag_dimension
        self.learnable_diag_1 = nn.Parameter(torch.FloatTensor(in_features))  # in_features

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.learnable_diag_1, mean=0, std=0)

    def forward(self, input, l_sym):

        e1 = torch.spmm(l_sym, input)
        alpha = torch.diag(self.learnable_diag_1)
        e2 = torch.mm(e1, alpha)
        e4 = torch.sub(input, e2)
        output = e4

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Adagnn_with_weight(nn.Module):

    def __init__(self, diag_dimension, in_features, out_features, bias=True):
        super(Adagnn_with_weight, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diag_dimension = diag_dimension
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.learnable_diag_1 = nn.Parameter(torch.FloatTensor(in_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.normal_(self.learnable_diag_1, mean=0, std=0.01)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, l_sym):
        e1 = torch.spmm(l_sym, input)
        alpha = torch.diag(self.learnable_diag_1)
        e2 = torch.mm(e1, alpha + torch.eye(self.in_features, self.in_features).cuda())
        e4 = torch.sub(input, e2)
        e5 = torch.mm(e4, self.weight)
        output = e5

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
# -------------------------------------------------------------------
# layers for ACM-GCN, modified from https://github.com/RecklessRonan/GloGNN/blob/master/large-scale/models.py
# -------------------------------------------------------------------

class ACMGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, variant=False):
        super(ACMGraphConvolution, self).__init__()
        self.in_features, self.out_features, self.variant = in_features, out_features, variant
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        if torch.cuda.is_available():
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features).cuda()), Parameter(
                torch.FloatTensor(in_features, out_features).cuda()), Parameter(torch.FloatTensor(in_features, out_features).cuda())
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1).cuda(
            )), Parameter(torch.FloatTensor(out_features, 1).cuda()), Parameter(torch.FloatTensor(out_features, 1).cuda())
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(1, 1).cuda(
            )), Parameter(torch.FloatTensor(1, 1).cuda()), Parameter(torch.FloatTensor(1, 1).cuda())

            self.att_vec = Parameter(torch.FloatTensor(3, 3).cuda())

        else:
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features)), Parameter(
                torch.FloatTensor(in_features, out_features)), Parameter(torch.FloatTensor(in_features, out_features))
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1)), Parameter(
                torch.FloatTensor(out_features, 1)), Parameter(torch.FloatTensor(out_features, 1))
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(
                1, 1)), Parameter(torch.FloatTensor(1, 1)), Parameter(torch.FloatTensor(1, 1))

            self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt(self.att_vec_mlp.size(1))

        std_att_vec = 1. / math.sqrt(self.att_vec.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_low, output_high, output_mlp):
        T = 3
        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat([torch.mm((output_low), self.att_vec_low), torch.mm(
            (output_high), self.att_vec_high), torch.mm((output_mlp), self.att_vec_mlp)], 1)), self.att_vec)/T, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, input, adj_low, adj_high):
        output = 0

        if self.variant:
            output_low = (torch.spmm(adj_low, F.relu(
                torch.mm(input, self.weight_low))))
            output_high = (torch.spmm(adj_high, F.relu(
                torch.mm(input, self.weight_high))))
            output_mlp = F.relu(torch.mm(input, self.weight_mlp))
        else:
            output_low = F.relu(torch.spmm(
                adj_low, (torch.mm(input, self.weight_low))))
            output_high = F.relu(torch.spmm(
                adj_high, (torch.mm(input, self.weight_high))))
            output_mlp = F.relu(torch.mm(input, self.weight_mlp))

        self.att_low, self.att_high, self.att_mlp = self.attention(
            (output_low), (output_high), (output_mlp))
        
        return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)
       

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

