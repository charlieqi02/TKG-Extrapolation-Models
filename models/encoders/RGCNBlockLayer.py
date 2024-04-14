import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNBlockLayer(nn.Module):
    def __init__(self, rank, num_rels):
        super().__init__()
        self.rank = rank
        self.num_rels = num_rels * 2
        self.submat = 2
        self.diag = (self.rank // 2) * (self.submat**2)
        self.activation = F.rrelu

        # relation parameter
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.diag))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))


    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                        -1, self.submat, self.submat)    # [edge_num, submat, submat]
        node = edges.src['h'].view(-1, 1, self.submat)   # [edge_num * diag, 1, submat]->
        msg = torch.bmm(node, weight).view(-1, self.rank)   # [edge_num, rank]
        return {'msg': msg}


    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

 
    def forward(self, g):
        self.propagate(g)  # GCN
        
        # apply bias and activation
        g.ndata['h'] = self.activation(g.ndata['h'])
        
        return g.ndata.pop('h')