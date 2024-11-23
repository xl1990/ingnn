import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch.nn import Parameter, init

import numpy as np
import scipy.sparse
from tqdm import tqdm
import faiss
from utils import norm_edges
import time
import math

from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops, get_laplacian, degree
from scipy.special import comb

from layers import Adagnn_without_weight, Adagnn_with_weight, ACMGraphConvolution


# ----------------------------------------------------------------
# Codes for aggregated neighborhood feature module
# ----------------------------------------------------------------
class AGGR(nn.Module):
    """module for extracting aggregated neighborhood feature"""
    def __init__(self, power=2):
        super(AGGR, self).__init__()
        self.power = power
    
    def reset_parameters(self):
        pass
    
    def forward(self,x,A, training_policy=False):
        hX = x
        for i in range(self.power-1):
            hX = A@hX+x
        return A@hX

# ----------------------------------------------------------------
# Codes for structure feature module
# ----------------------------------------------------------------
class STRC(nn.Module):
    """module for extracting structure feature"""
    def __init__(self, in_channels, out_channels, power=2):
        super(STRC, self).__init__()
        self.power = power
        self.W = nn.Parameter(torch.empty((in_channels, out_channels)))
        self.bn_list = nn.ModuleList()
        for i in range(power):
            self.bn_list.append(nn.BatchNorm1d(out_channels))
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        for bn in self.bn_list:
            bn.reset_parameters()
    
    def forward(self,A, training_policy=False):
        xAs = []      
        W = self.W
        for i in range(self.power):
            bn = self.bn_list[i]
            if training_policy:
                bn.eval()
            else:
                bn.train()
                
            temp = matmul(A,W)
            W = bn(temp)
            xAs.append(W)
        return torch.stack(xAs).mean(dim=0)
    
class LINKX_AGGR(nn.Module):
    """ LINKX with aggregated node feature 
        a = MLP_1(A), x = MLP_2(X), x_aggr=(A^k)*x, MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1, power=2):
        super(LINKX_AGGR, self).__init__()	
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(3*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout
        
        self.aggr = AGGR(power)
        self.hp_list = ['hidden_channels','num_layers','power']

    def reset_parameters(self):
        self.mlpA.reset_parameters()
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()

    def forward(self, data):
        m = data.graph['num_nodes']
        feat_dim = data.graph['node_feat']
        row, col = data.graph['edge_index']
        row = row-row.min()
        A = SparseTensor(row=row, col=col,
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()
        
        edge_index, norm = norm_edges(data.graph['edge_index'], data.graph['node_feat'])
        A_norm = SparseTensor(row=row, col=col, value = norm,
                         sparse_sizes=(m, self.num_nodes)).to_torch_sparse_coo_tensor()

#         s = time.time()
        xA = self.mlpA(A, input_tensor=True)
#         t = time.time()
#         print(t-s)
        
        xX = self.mlpX(data.graph['node_feat'], input_tensor=True)
        hX = self.aggr(xX,A_norm)
        
        x = torch.cat((xA, xX, hX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX + hX)
        x = self.mlp_final(x, input_tensor=True)

        return x

class LINKX(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()	
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	

    def forward(self, data):	
        m = data.graph['num_nodes']	
        feat_dim = data.graph['node_feat']	
        row, col = data.graph['edge_index']	
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()

#         s = time.time()
        xA = self.mlpA(A, input_tensor=True)
#         t = time.time()
#         print(t-s)
        
        xX = self.mlpX(data.graph['node_feat'], input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        return x

class LINK(nn.Module):
    """ logistic regression on adjacency matrix """
    
    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, data):
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        return logits

class LINK_Concat(nn.Module):	
    """ concate A and X as joint embeddings i.e. MLP([A;X])"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=True):	
        super(LINK_Concat, self).__init__()	
        self.mlp = MLP(in_channels + num_nodes, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels	
        self.cache = cache
        self.x = None

    def reset_parameters(self):	
        self.mlp.reset_parameters()	

    def forward(self, data):	
        if (not self.cache) or (not isinstance(self.x, torch.Tensor)):
                N = data.graph['num_nodes']	
                feat_dim = data.graph['node_feat']	
                row, col = data.graph['edge_index']	
                col = col + self.in_channels	
                feat_nz = data.graph['node_feat'].nonzero(as_tuple=True)	
                feat_row, feat_col = feat_nz	
                full_row = torch.cat((feat_row, row))	
                full_col = torch.cat((feat_col, col))	
                value = data.graph['node_feat'][feat_nz]	
                full_value = torch.cat((value, 	
                                torch.ones(row.shape[0], device=value.device)))	
                x = SparseTensor(row=full_row, col=full_col,	
                         sparse_sizes=(N, N+self.in_channels)	
                            ).to_torch_sparse_coo_tensor()	
                if self.cache:
                    self.x = x
        else:
                x = self.x
        logits = self.mlp(x, input_tensor=True)
        return logits


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=True) 

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        x = self.conv(x, edge_index)
        return x


class SGCMem(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ lower memory version (if out_channels < in_channels)
        takes weight multiplication first, then propagate
        takes hops power of the normalized adjacency
        """
        super(SGCMem, self).__init__()

        self.lin = nn.Linear(in_channels, out_channels)
        self.hops = hops

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        x = self.lin(x)
        n = data.graph['num_nodes']
        edge_weight=None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False,
                 dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index

        for _ in range(self.hops):
            x = matmul(adj_t, x)
        
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x
    
class GCN_STRC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True,power=2):
        super(GCN_STRC, self).__init__()

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        
        # layers for structure feature
        self.strc_feature_embed = STRC(num_nodes, out_channels, power=power)
        
        # policy for merging the node feature and structure feature
        self.policy = nn.Parameter(torch.empty((2)))
        self.activated_policy = None
        
        self.hp_list = ['hidden_channels','power']

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
        self.strc_feature_embed.reset_parameters()
        nn.init.zeros_(self.policy)
        
    def freeze_policy(self):
        self.policy.requires_grad = False
        return
        
    def unfreeze_policy(self):
        self.policy.requires_grad = True
        return

    def forward(self, data, training_policy=False):
        num_nodes = data.graph['num_nodes']
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        row, col = edge_index
        A = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)) #.to_torch_sparse_coo_tensor()

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        
        xA = self.strc_feature_embed(A)
        pp = F.softmax(self.policy, dim=-1)
        self.activated_policy = pp.detach().cpu().numpy()
        
        x = pp[0]*x + pp[1]*xA
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, sampling=False, add_self_loops=True):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.elu 
        self.sampling = sampling
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data, adjs=None, x_batch=None):
        if not self.sampling:
            x = data.graph['node_feat']
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, data.graph['edge_index'])
                x = self.bns[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, data.graph['edge_index'])
        else:
            x = x_batch
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def inference(self, data, subgraph_loader):
        x_all = data.graph['node_feat'] 
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        total_edges = 0
        device = x_all.device
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class MultiLP(nn.Module):
    """ label propagation, with possibly multiple hops of the adjacency """
    
    def __init__(self, out_channels, alpha, hops, num_iters=50, mult_bin=False):
        super(MultiLP, self).__init__()
        self.out_channels = out_channels
        self.alpha = alpha
        self.hops = hops
        self.num_iters = num_iters
        self.mult_bin = mult_bin # handle multiple binary tasks
        
    def forward(self, data, train_idx):
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight=None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False)
            row, col = edge_index
            # transposed if directed
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False)
            edge_weight=None
            adj_t = edge_index

        y = torch.zeros((n, self.out_channels)).to(adj_t.device())
        if data.label.shape[1] == 1:
            # make one hot
            y[train_idx] = F.one_hot(data.label[train_idx], self.out_channels).squeeze(1).to(y)
        elif self.mult_bin:
            y = torch.zeros((n, 2*self.out_channels)).to(adj_t.device())
            for task in range(data.label.shape[1]):
                y[train_idx, 2*task:2*task+2] = F.one_hot(data.label[train_idx, task], 2).to(y)
        else:
            y[train_idx] = data.label[train_idx].to(y.dtype)
        result = y.clone()
        for _ in range(self.num_iters):
            for _ in range(self.hops):
                result = matmul(adj_t, result)
            result *= self.alpha
            result += (1-self.alpha)*y

        if self.mult_bin:
            output = torch.zeros((n, self.out_channels)).to(result.device)
            for task in range(data.label.shape[1]):
                output[:, task] = result[:, 2*task+1]
            result = output

        return result


class MixHopLayer(nn.Module):
    """ Our MixHop layer """
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x) ]
        for j in range(1,self.hops+1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)

class MixHop(nn.Module):
    """ our implementation of MixHop
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(hidden_channels*(hops+1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))

        self.convs.append(
            MixHopLayer(hidden_channels*(hops+1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels*(hops+1), out_channels)

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False,
                 dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)
        return x

class GCNJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()

        cached = False
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x

class GATJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, jk_type='max'):
        super(GATJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, hidden_channels, heads=heads))

        self.dropout = dropout
        self.activation = F.elu # note: uses elu

        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels*heads, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels*heads*num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels*heads, out_channels)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    # def forward(self, x, adj_t, adj_t2):
    #     x1 = matmul(adj_t, x)
    #     x2 = matmul(adj_t2, x)
    #     return torch.cat([x1, x2], dim=1)
    
    def forward(self, x, adj_t, adj_t2=None):
        x1 = matmul(adj_t, x)
        
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    """ our implementation """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)


        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)
        
        self.hp_list = ['hidden_channels','num_layers','dropout']

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)



    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x
    

        
    
class H2GCN_STRC(nn.Module):
    """ our implementation """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True):
        super(H2GCN_STRC, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)


        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )


        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )
                

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        
        # self.final_project = nn.Linear(last_dim, out_channels)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)
        
        # layers for structure feature
        self.strc_feature_embed = STRC(num_nodes, last_dim, power=2*num_layers)
        
        # policy for merging the node feature and structure feature
        self.policy = nn.Parameter(torch.empty((2)))
        self.activated_policy = None
        
        self.hp_list = ['hidden_channels','num_layers','dropout']

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
        self.strc_feature_embed.reset_parameters()
        nn.init.zeros_(self.policy)
        
    def freeze_policy(self):
        self.policy.requires_grad = False
        return
        
    def unfreeze_policy(self):
        self.policy.requires_grad = True
        return

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        self.adj_t_ori = adj_t.to(edge_index.device)
        self.adj_t2_ori = adj_t2.to(edge_index.device)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)



    def forward(self, data, training_policy=False):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        adj_t_ori = self.adj_t_ori
        adj_t2_ori = self.adj_t2_ori
        
        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training and not training_policy)
        
        # set to eval mode when training policy
        for bn in self.bns:
            if training_policy:
                bn.eval()
            else:
                bn.train()
                
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training and not training_policy)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training and not training_policy)
        xs.append(x)
        # print([x.shape for x in xs])

        x = self.jump(xs)
        # print(x.shape)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training and not training_policy)
        
        # x_adj = self.strc_feature_embed(self.adj_t_ori)
        x_adj = self.strc_feature_embed(adj_t)
        
        pp = F.softmax(self.policy, dim=-1)
        self.activated_policy = pp.detach().cpu().numpy()
        x = pp[0]*x + pp[1]*x_adj
        
        x = self.final_project(x)
        return x


class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dprate=.0, dropout=.5, K=10, alpha=.1, num_layers=3):
        super(APPNP_Net, self).__init__()
        
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = APPNP(K, alpha)

        self.dprate = dprate
        self.dropout = dropout
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x





class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Init='Random', dprate=.0, dropout=.5, K=10, alpha=.1, Gamma=None, num_layers=3):
        super(GPRGNN, self).__init__()
        
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.5):
        super(GCNII, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = dropout
        
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
    
class OGNN(nn.Module):
    """ 
    OGNN (Omnipotent Graph Neural Network)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, power1, power2, dropout=.5, tau_decay=0.99, ens='111'):
        super(OGNN, self).__init__()
        self.linX = nn.Linear(in_channels,hidden_channels)
        
        self.wA = Parameter(torch.empty((num_nodes, hidden_channels)))
        self.bn_list = nn.ModuleList()
        for i in range(power2):
            self.bn_list.append(nn.BatchNorm1d(hidden_channels))
        
        self.policy = Parameter(torch.empty((3)))
        self.pred = nn.Linear(hidden_channels,out_channels)
        
        self.num_nodes = num_nodes
        self.power1 = power1
        self.power2 = power2
        self.dropout = dropout
        self.description = "linkdex_df concatenate features\n"
        
        self.alpha = -1
        
        self.scales = torch.zeros(3)
        self.activated_policy = None
        
        self.hp_list = ['hidden_channels','power1','power2','normalize_feature']

    def reset_parameters(self):
        self.linX.reset_parameters()
        nn.init.kaiming_uniform_(self.wA, a=math.sqrt(5))
        for bn in self.bn_list:
            bn.reset_parameters()
        nn.init.zeros_(self.policy)
        self.pred.reset_parameters()
    
    def reset_parameters_without_policy(self):
        self.linX.reset_parameters()
        nn.init.kaiming_uniform_(self.wA, a=math.sqrt(5))
        for bn in self.bn_list:
            bn.reset_parameters()
        self.pred.reset_parameters()
        
    def forward(self, data,training_policy=False):
        num_nodes = data.graph['num_nodes']
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        
        edge_index, norm = self.norm_edges(edge_index, x)
        row, col = edge_index
        row = row-row.min()
        
        A_ori = SparseTensor(row=row, col=col, #value = norm,
                         sparse_sizes=(num_nodes, self.num_nodes)).to_torch_sparse_coo_tensor()
        A_norm = SparseTensor(row=row, col=col, value = norm,
                         sparse_sizes=(num_nodes, self.num_nodes)).to_torch_sparse_coo_tensor()
        # A_ori = A_norm
        
        x = F.dropout(x,p=self.dropout,training=self.training and not training_policy)
        
        # ego node feature
        xX = self.linX(x)
        

        pp = F.softmax(self.policy[:3], dim=-1)
        self.activated_policy = pp.detach().cpu().numpy()
        
        # aggregated neighborhood feature
        hX = xX
        for i in range(self.power1):
            hX = A_norm@hX+xX
        
        # structure features
        xAs = []      
        w2 = self.wA
        for i in range(self.power2):
            bn = self.bn_list[i]
            if training_policy:
                bn.eval()
            else:
                bn.train()
            
            temp = F.linear(A_ori,w2.t())
            w2 = bn(temp)
            xAs.append(w2)
        hA = torch.stack(xAs).sum(dim=0)
        
        x = pp[0]*xX + pp[1]*hX + pp[2]*hA 
        
        x = F.dropout(x,p=self.dropout,training=self.training and not training_policy)
        x = F.relu(x)
            
        x = self.pred(x)
        return x
        
    def norm_edges(self,edge_index, x):
        # edge_index,_ = add_self_loops(edge_index)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        try:
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        except:
            print(deg_inv_sqrt)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return edge_index, norm
    
    def freeze_policy(self):
        self.policy.requires_grad = False
        # self.useless_param.requires_grad = False
        return
        
    def unfreeze_policy(self):
        self.policy.requires_grad = True
        # self.useless_param.requires_grad = True
        return

# ----------------------------------------------------------------
# Codes for DAGNN
# ----------------------------------------------------------------
def gcn_norm_dagnn(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = nn.Linear(num_classes, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm_dagnn(edge_index, edge_weight, x.size(0), dtype=x.dtype)
        


        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()
    
class DAGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels,power,dropout):
        super(DAGNN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = Prop(out_channels, power)
        self.dropout = dropout
        self.hp_list = ['hidden_channels','power','normalize_feature']

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        num_nodes = data.graph['num_nodes']
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x
        # return F.log_softmax(x, dim=1)

# ----------------------------------------------------------------
# Codes for Bern Net
# ----------------------------------------------------------------
class Bern_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.temp = nn.Parameter(torch.Tensor(self.K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index,edge_weight=None):
        TEMP=F.relu(self.temp)

        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
        #2I-L
        edge_index2, norm2=add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
            x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
            tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
            x=tmp[self.K-i-1]
            x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
            for j in range(i):
                x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

            out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x
        return out
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
    
class BernNet(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels,out_channels,power=10,dropout=0.5,dprate=0.5):
        super(BernNet, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.m = nn.BatchNorm1d(out_channels)
        self.prop1 = Bern_prop(power)

        self.dprate = dropout
        self.dropout = dprate
        
        self.hp_list = ['hidden_channels','power','dropout','dprate']

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.m.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            # return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            # return F.log_softmax(x, dim=1)
        return x

# ----------------------------------------------------------------
# Codes for AdaGNN
# ----------------------------------------------------------------
class AdaGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, num_layers, dropout, mode='s'):
        super(AdaGNN, self).__init__()

        self.should_train_1 = Adagnn_with_weight(num_nodes, in_channels, hidden_channels)
        assert num_layers - 2 >= 0
        self.hidden_layers = nn.ModuleList([
            Adagnn_without_weight(in_channels, hidden_channels, hidden_channels, bias=False)
            for i in range(num_layers - 2)
        ])
        self.should_train_2 = Adagnn_with_weight(num_nodes, hidden_channels, out_channels)
        self.dropout = dropout
        
        self.l_sym = None
        self.mode = mode # normalization mode
        
        self.hp_list = ['num_layers','hidden_channels','dropout','mode']
    
    def reset_parameters(self):
        self.should_train_1.reset_parameters()
        for hidden_layer in self.hidden_layers:
            hidden_layer.reset_parameters()
        self.should_train_2.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        if self.l_sym is None:
            self.l_sym = self.gen_l_sym(data)
        l_sym = self.l_sym
        
        x = F.relu(self.should_train_1(x, l_sym))  # .relu
        x = F.dropout(x, self.dropout, training=self.training)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, l_sym)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.should_train_2(x, l_sym)  # + res1
        return x
    
    def gen_l_sym(self, data):
        num_nodes = data.graph['num_nodes']
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']

        edge_index,_ = add_self_loops(edge_index)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        A = SparseTensor(row=row, col=col, 
                         sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
        D = SparseTensor(row=torch.LongTensor([i for i in range(num_nodes)]).to(deg.device), 
                         col=torch.LongTensor([i for i in range(num_nodes)]).to(deg.device), 
                         value=deg,
                         sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
        L = D-A

        if self.mode=='s':
            deg_inv_sqrt = deg.pow(-0.5)
            try:
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            except:
                print(deg_inv_sqrt)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        elif self.mode=='r':
            deg_inv_sqrt = deg.pow(-1)
            try:
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            except:
                print(deg_inv_sqrt)
            norm = deg_inv_sqrt[row] 
        else:
            raise sys.exit('Wrong mode')

        D_norm = SparseTensor(row=row, col=col, value = norm,
                              sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
        return D_norm*L
    
# ----------------------------------------------------------------
# Codes for ACMGCN
# ----------------------------------------------------------------
    
class ACMGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, variant=False):
        super(ACMGCN, self).__init__()
        self.gcns = nn.ModuleList()

        self.gcns.append(ACMGraphConvolution(in_channels, hidden_channels, variant=variant))
        self.gcns.append(ACMGraphConvolution(hidden_channels, out_channels, variant=variant))

        self.dropout = dropout
        self.hp_list = ['hidden_channels','dropout','variant']

    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()

    def forward(self, data):
        num_nodes = data.graph['num_nodes']
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']

        edge_index,_ = add_self_loops(edge_index)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)

        deg_inv_sqrt = deg.pow(-1)
        try:
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        except:
            print(deg_inv_sqrt)
        norm = deg_inv_sqrt[row] 
        adj_low = SparseTensor(row=row, col=col, value=norm,
                               sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
        I = SparseTensor(row=torch.LongTensor([i for i in range(num_nodes)]).to(x.device), 
                         col=torch.LongTensor([i for i in range(num_nodes)]).to(x.device), 
                         value=torch.ones_like(deg),
                         sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
        adj_high = I-adj_low
        
        x = F.dropout(x, self.dropout, training=self.training)

        fea = (self.gcns[0](x, adj_low, adj_high))

        fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
        fea = self.gcns[-1](fea, adj_low, adj_high)
        return fea
    
# ----------------------------------------------------------------
# Codes for GLOGNN
# ----------------------------------------------------------------    
    
class GLOGNN(nn.Module):
    def __init__(self,  in_channels, hidden_channels, out_channels, num_nodes, num_layers, dropout, 
                 alpha, beta, gamma, delta, norm_func_id,  orders, orders_func_id, device):
        
        super(GLOGNN, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = nn.Linear(num_nodes, hidden_channels)
        # self.bn1 = nn.BatchNorm1d(nhid)
        # self.bn2 = nn.BatchNorm1d(nhid)
        self.out_channels = out_channels
        self.dropout = dropout
        self.alpha = torch.tensor(alpha).to(device)
        self.beta = torch.tensor(beta).to(device)
        self.gamma = torch.tensor(gamma).to(device)
        self.delta = torch.tensor(delta).to(device)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.orders = orders
        self.device = device
        self.class_eye = torch.eye(self.out_channels).to(device)
        self.orders_weight = Parameter(
            (torch.ones(orders, 1) / orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix = Parameter(
            torch.DoubleTensor(out_channels, orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix2 = Parameter(
            torch.DoubleTensor(orders, orders).to(device), requires_grad=True
        )
        self.diag_weight = Parameter(
            (torch.ones(out_channels, 1) / out_channels).to(device), requires_grad=True
        )
        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3
        
        self.hp_list = ['hidden_channels','num_layers','dropout','alpha','beta','gamma','delta','orders']


    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.orders_weight = Parameter(
            (torch.ones(self.orders, 1) / self.orders).to(self.device), requires_grad=True
        )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.diag_weight = Parameter(
            (torch.ones(self.out_channels, 1) / self.out_channels).to(self.device), requires_grad=True
        )

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                           sparse_sizes=(data.graph['num_nodes'], data.graph['num_nodes'])).to_torch_sparse_coo_tensor()

        
        # xd = F.dropout(x, self.dropout, training=self.training)
        # adjd = F.dropout(adj, self.dropout, training=self.training)
        xX = self.fc1(x)
        # x = self.bn1(x)
        xA = self.fc4(adj)
        x = F.relu(self.delta * xX + (1-self.delta) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        # x = self.bn2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        h0 = x
        for _ in range(self.num_layers):
            x = self.norm(x, h0, adj)
        return x

    def norm_func1(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1.0 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def norm_func2(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def order_func1(self, x, res, adj):
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        orders_para = torch.transpose(orders_para, 0, 1)
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders
    
    
# ----------------------------------------------------------------
# OGNN without structure feature, for ablation
# ----------------------------------------------------------------  
    
class OGNN_NO_STRC(nn.Module):
    """ 
    OGNN (Omnipotent Graph Neural Network) without structure feature
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, power1, power2, dropout=.5, tau_decay=0.99, ens='111'):
        super(OGNN_NO_STRC, self).__init__()
        self.linX = nn.Linear(in_channels,hidden_channels)
        
        self.wA = Parameter(torch.empty((num_nodes, hidden_channels)))
        self.bn_list = nn.ModuleList()
        for i in range(power2):
            self.bn_list.append(nn.BatchNorm1d(hidden_channels))
        
        self.policy = Parameter(torch.empty((3)))
        self.pred = nn.Linear(hidden_channels,out_channels)
        
        self.num_nodes = num_nodes
        self.power1 = power1
        self.power2 = power2
        self.dropout = dropout
        
        self.alpha = -1
        
        self.scales = torch.zeros(3)
        self.activated_policy = None
        
        self.hp_list = ['hidden_channels','power1','power2','normalize_feature']

    def reset_parameters(self):
        self.linX.reset_parameters()
        nn.init.kaiming_uniform_(self.wA, a=math.sqrt(5))
        for bn in self.bn_list:
            bn.reset_parameters()
        nn.init.zeros_(self.policy)
        self.pred.reset_parameters()
    
    def reset_parameters_without_policy(self):
        self.linX.reset_parameters()
        nn.init.kaiming_uniform_(self.wA, a=math.sqrt(5))
        for bn in self.bn_list:
            bn.reset_parameters()
        self.pred.reset_parameters()
        
    def forward(self, data,training_policy=False):
        num_nodes = data.graph['num_nodes']
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        
        edge_index, norm = self.norm_edges(edge_index, x)
        row, col = edge_index
        row = row-row.min()
        
        # A_ori = SparseTensor(row=row, col=col, #value = norm,
        #                  sparse_sizes=(num_nodes, self.num_nodes)).to_torch_sparse_coo_tensor()
        A_norm = SparseTensor(row=row, col=col, value = norm,
                         sparse_sizes=(num_nodes, self.num_nodes)).to_torch_sparse_coo_tensor()
        # A_ori = A_norm
        
        x = F.dropout(x,p=self.dropout,training=self.training and not training_policy)
        
        # ego node feature
        xX = self.linX(x)
        

        pp = F.softmax(self.policy[:2], dim=-1)
        self.activated_policy = pp.detach().cpu().numpy()
        
        # aggregated neighborhood feature
        hX = xX
        for i in range(self.power1):
            hX = A_norm@hX+xX
        
#         # structure features
#         xAs = []      
#         w2 = self.wA
#         for i in range(self.power2):
#             bn = self.bn_list[i]
#             if training_policy:
#                 bn.eval()
#             else:
#                 bn.train()
            
#             temp = F.linear(A_ori,w2.t())
#             w2 = bn(temp)
#             xAs.append(w2)
#         hA = torch.stack(xAs).sum(dim=0)
        
        x = pp[0]*xX + pp[1]*hX
        
        x = F.dropout(x,p=self.dropout,training=self.training and not training_policy)
        x = F.relu(x)
            
        x = self.pred(x)
        return x
        
    def norm_edges(self,edge_index, x):
        # edge_index,_ = add_self_loops(edge_index)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        try:
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        except:
            print(deg_inv_sqrt)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return edge_index, norm
    
    def freeze_policy(self):
        self.policy.requires_grad = False
        # self.useless_param.requires_grad = False
        return
        
    def unfreeze_policy(self):
        self.policy.requires_grad = True
        # self.useless_param.requires_grad = True
        return