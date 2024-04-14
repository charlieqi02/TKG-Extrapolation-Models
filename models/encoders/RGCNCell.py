import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNCell(nn.Module):
    def __init__(self, sizes, rank, num_layers=1, self_loop=False, dropout=0):
        super().__init__()
        self.sizes = sizes
        self.rank = rank
        self.self_loop = self_loop
        
        # create rgcn layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            h2h = UnionRGCNLayer(self.sizes, self.rank, self.self_loop, dropout)
            self.layers.append(h2h)

    def forward(self, g, ent_emb, rel_emb):
        g.ndata['h'] = ent_emb
        for layer in self.layers:
            layer(g, rel_emb)
        return g.ndata.pop('h')


class UnionRGCNLayer(nn.Module):
    def __init__(self, sizes, rank, self_loop=False, dropout=0.0):
        super(UnionRGCNLayer, self).__init__()
        self.num_ents = sizes[0]
        self.rank = rank
        self.self_loop = self_loop
        self.activation = F.rrelu

        self.weight_neighbor = nn.Parameter(torch.Tensor(self.rank, self.rank))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(self.rank, self.rank))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.alone_loop_weight = nn.Parameter(torch.Tensor(self.rank, self.rank))
            nn.init.xavier_uniform_(self.alone_loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)


    def forward(self, g, rel_emb):
        self.rel_emb = rel_emb
        node_repr = g.ndata['h']
        
        # calculate the neighbor message with weight_neighbor
        # g.ndata['h'] is changed
        self.propagate(g)
        
        if self.self_loop:
            # True -> nodes have neighbor; False -> node don't have neighbor
            masked_index = torch.masked_select(g.ndata['id'].squeeze(), g.in_degrees(range(self.num_ents)) > 0)
            loop_message = torch.mm(node_repr, self.alone_loop_weight)
            loop_message[masked_index, :] = torch.mm(node_repr, self.loop_weight)[masked_index, :]
            node_repr = g.ndata['h'] + loop_message
        else: 
            node_repr = g.ndata['h']
       
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr


    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def msg_func(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['type'])
        node = edges.src['h']
        msg = node + relation
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}