import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRUCell

from utils.gcn import build_sub_graph, inverse_snapshot

from .base import TKGModel
from .decoders import ConvTransE
from .encoders import RGCNBlockLayer, RGCNCell, TimeGate

REGCN_MODELS = ['SimREGCN']


class SimREGCN(TKGModel):
    def __init__(self, args):
        super(SimREGCN, self).__init__(args.sizes, args.rank, args.dropout, args.gpu)
        self.num_layers = args.num_rgcn_layers
        self.self_loop = args.self_loop
        self.use_static = args.use_static
        self.channel = args.channel
        self.kernel_size = args.kernel_size
               
        if self.use_static:
            self.static_sizes = args.aux['static_size']      # (num words, num static rels) non-inversed
            self.wrd_emb = nn.Embedding(self.static_sizes[0], self.rank)
            self.static_rgcn = RGCNBlockLayer(self.rank, self.static_sizes[1])
        self.rgcn = RGCNCell(self.sizes, self.rank, self.num_layers, self.self_loop, self.dropout)
        self.update = GRUCell(self.rank, self.rank)
        self.hr_mix = ConvTransE(self.rank, self.channel, self.kernel_size, self.dropout)


    def get_embeddings(self, snapshots, auxiliaries):
        """
        auxiliaries options:
            'static': dgl.DGLGraph() containing static graph information
        """
        ent_embs, rel_embs, extra_embs = {}, {}, {}
        
        # static
        if self.use_static:
            static_graph = auxiliaries['static']
            static_graph.ndata['h'] = torch.cat((self.ent_emb.weight, self.wrd_emb.weight), 0)
            h = F.normalize(self.static_rgcn.forward(static_graph)[:self.sizes[0], :])      # static embedding
            extra_embs['static_emb'] = h
        else:
            h = F.normalize(self.ent_emb.weight)
        
        # recurrent (dynamic)
        r = F.normalize(self.rel_emb.weight)
        rel_embs['rel_emb'] = r
        ent_embs['his_embs'] = []
        for snapshot in snapshots:
            snapshot = inverse_snapshot(self.sizes[1], snapshot)
            snap_graph = build_sub_graph(self.sizes[0], snapshot, self.gpu)

            h_input = self.rgcn.forward(snap_graph, h, r)
            h_input = F.normalize(h_input)
            
            h = self.update.forward(h_input, h)
            h = F.normalize(h)
            ent_embs['his_embs'].append(h)
                    
        return ent_embs, rel_embs, extra_embs
    
    
    def get_queries(self, queries, ent_embs, rel_embs, extra_embs):
        qry_embs, cdd_embs = {}, {}
        h = ent_embs['his_embs'][-1]
        r = rel_embs['rel_emb']
        all_queries = inverse_snapshot(self.sizes[1], queries, self.gpu)
        
        qry_emb, cdd_emb = self.hr_mix.forward(h, r, all_queries)
        qry_embs['qry_emb'] = qry_emb
        cdd_embs['cdd_emb'] = cdd_emb
        return all_queries[:, 2], qry_embs, cdd_embs
    
    
    def score(self, qry_embs, cdd_embs):
        qry_emb = qry_embs['qry_emb']
        cdd_emb = cdd_embs['cdd_emb']
        scores = qry_emb @ cdd_emb.t()
        return scores