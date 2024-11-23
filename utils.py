import torch
import faiss
import faiss.contrib.torch_utils
from torch_geometric.utils import add_self_loops, degree
import sys

def knn_faiss(x_b,x_q,k,res):
    n,d = x_b.shape
    with torch.no_grad():
        gpu_index = faiss.GpuIndexFlatL2(res, d)
        gpu_index.add(x_b)
        distance, index = gpu_index.search(x_q, k)

        return torch.stack([torch.LongTensor([[i]*k for i in range(n)]).to(x_b.device), index.long()]).view(2,-1)  
    
def norm(x):
    eps = 1e-16
    mean = x.mean(dim=-1,keepdim=True)
    std = x.std(dim=-1,keepdim=True)
    
    return (x-mean)/(std+eps)

# def norm(x):
#     eps = 1e-16
#     return x/(torch.sqrt(x.square().sum(-1,keepdim=True))+eps)

def norm_edges(edge_index, x, self_loop=False):
    if self_loop:
        edge_index,_ = add_self_loops(edge_index)
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)

    deg_inv_sqrt = deg.pow(-0.5)
    try:
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    except:
        print(deg_inv_sqrt)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return edge_index, norm


