"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.euclidean import givens_reflection, givens_rotations
from utils.hyperbolic import expmap0, hyp_distance_multi_c, mobius_add, project
from utils.gcn import build_sub_graph, inverse_snapshot

from .base import TKGModel

HYP_MODELS = ["RotH", "RefH", "AttH"]


class BaseH(TKGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gpu)
        self.dtype = args.dtype
        self.init_size = args.init_size
        self.multi_c = args.multi_c
        
        self.data_type = torch.double if self.dtype == "double" else torch.float
        self.bh = nn.Embedding(self.sizes[0], 1)
        self.bh.weight.data = torch.zeros((self.sizes[0], 1), dtype=self.data_type)
        self.bt = nn.Embedding(self.sizes[0], 1)
        self.bt.weight.data = torch.zeros((self.sizes[0], 1), dtype=self.data_type)
        
        self.rel_diag = nn.Embedding(self.sizes[1]*2, self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1]*2, self.rank), dtype=self.data_type) - 1.0
        self.ent_emb.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel_emb.weight.data = self.init_size * torch.randn((self.sizes[1]*2, self.rank*2), dtype=self.data_type)
        
        c_num = self.sizes[1]*2 if self.multi_c else 1
        c_init = torch.ones((c_num, 1), dtype=self.data_type) 
        self.c = nn.Parameter(c_init, requires_grad=True)
    
    def get_embeddings(self, snapshots, auxiliaries):
        ent_embs, rel_embs, extra_embs = {}, {}, {}
        return ent_embs, rel_embs, extra_embs
    
    def get_queries(self, queries, ent_embs, rel_embs, extra_embs):
        qry_embs, cdd_embs = {}, {}
        all_queries = inverse_snapshot(self.sizes[1], queries, self.gpu)
        
        final_q, c = self.get_qhr(all_queries)
        
        qry_embs['qry_emb'] = final_q
        qry_embs['h_bias']  = self.bh(all_queries[:, 0])
        qry_embs['multi_c'] = c
        cdd_embs['cdd_emb'] = self.ent_emb.weight
        cdd_embs['t_bias']  = self.bt.weight
        return all_queries[:, 2], qry_embs, cdd_embs

    def score(self, qry_embs, cdd_embs):
        qry_emb = qry_embs['qry_emb']
        h_bias  = qry_embs['h_bias']
        c = qry_embs ['multi_c']
        cdd_emb = cdd_embs['cdd_emb']
        t_bias  = cdd_embs['t_bias']
        
        score = self.similarity_score(qry_emb, cdd_emb, c)       
        score += h_bias + t_bias.t()
        return score

    def similarity_score(self, qry_emb, cdd_emb, c):
        """Compute similarity scores in embedding space."""
        return - hyp_distance_multi_c(qry_emb, cdd_emb, c, True) ** 2


class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_qhr(self, all_queries):
        """Compute embedding and biases of queries."""   
        r1_emb, r2_emb = torch.chunk(self.rel_emb(all_queries[:, 1]), 2, dim=1)
        # get h + r1
        c = F.softplus(self.c[all_queries[:, 1]])
        h_emb = self.ent_emb(all_queries[:, 0])
        h_embH = expmap0(h_emb, c)
        r1_embH = expmap0(r1_emb, c)
        hr1_emb = project(mobius_add(h_embH, r1_embH, c), c)
        
        # get rot_q
        rot_mat = self.rel_diag(all_queries[:, 1])
        rot_q = givens_rotations(rot_mat, hr1_emb)
        
        # q_rot + r2
        r2_embH = expmap0(r2_emb)
        final_q = mobius_add(rot_q, r2_embH, c)
        return final_q, c


class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def get_qhr(self, all_queries):
        """Compute embedding and biases of queries."""        
        # get q_ref
        c = F.softplus(self.c[all_queries[:, 1]])
        h_emb = self.ent_emb(all_queries[:, 0])
        ref_mat = self.rel_diag(all_queries[:, 1])
        ref_q = givens_reflection(ref_mat, h_emb)
        ref_qH = expmap0(ref_q, c)
        
        # get q_ref + r
        r_emb, _ = torch.chunk(self.rel_emb(all_queries[:, 1]), 2, dim=1)
        r_embH = expmap0(r_emb, c)
        final_q = project(mobius_add(ref_qH, r_embH, c), c)
        return final_q, c


class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1]*2, self.rank*2)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1]*2, self.rank*2), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1]*2, self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1]*2, self.rank), dtype=self.data_type)
        self.softmax = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_qhr(self, all_queries):
        """Compute queries with Att(rot, ref, c) + r."""
        # get q_rot, q_ref
        c = F.softplus(self.c[all_queries[:, 1]])
        h_emb = self.ent_emb(all_queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(all_queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, h_emb).unsqueeze(1)
        ref_q = givens_reflection(ref_mat, h_emb).unsqueeze(1)
        cands = torch.cat([ref_q, rot_q], dim=1)
        
        # get Att(q_rot, q_ref, c)_H
        context_vec = self.context_vec(all_queries[:, 1]).unsqueeze(1)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=2, keepdim=True)
        att_weights = self.softmax(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        att_qH = expmap0(att_q, c)
        
        # get attq + r
        r_emb, _ = torch.chunk(self.rel_emb(all_queries[:, 1]), 2, dim=1)
        r_embH = expmap0(r_emb, c)     
        final_q = project(mobius_add(att_qH, r_embH, c), c)       
        return final_q, c
