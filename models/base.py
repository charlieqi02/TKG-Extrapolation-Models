"""Base Temporal Knowledge Graph embedding (Extrapoltaion) model."""
from abc import ABC, abstractmethod

import torch
from torch import nn

from utils.gcn import inverse_snapshot
from utils.test import get_hits, get_total_rank


class TKGModel(nn.Module, ABC):
    """Base Temporal Knowledge Graph Embedding (Extrapolation) model class.
       Input: original non-inversed np.array triples (info) and extra-info
       Output: Inference of the query triple.

    Attributes:
        sizes: Tuple[int, int] with (n_entities, n_relations) (not doubled)
        rank: integer for embedding dimension
        dropout: float for dropout rate
        ent_emb: torch.nn.Embedding with entity embeddings
        rel_emb: torch.nn.Embedding with relation embeddings (inversed - doubled)
    """

    def __init__(self, sizes, rank, dropout, gpu):
        """Initialize TKGModel."""
        super(TKGModel, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.ent_emb = nn.Embedding(sizes[0], rank)
        self.rel_emb = nn.Embedding(sizes[1]*2, rank)
        self.gpu = gpu

    @abstractmethod
    def get_embeddings(self, snapshots, auxiliaries):
        """Get encoded (allow multi-encoders) embeddings of entities, relations and etra_embeddings by KGs and Auxs.
        
        Args:
            snapshots: List of np.array triples with TKG snapshots
            auxiliaries: Dictionary mapping Strings to auxiliary information 
            
        Returns:
            ent_embs: Dictionary mapping encode-ent-content Strings to torch.Tensor (or Tensor List)
            rel_embs: Dictionary mapping encode-rel-content Strings to torch.Tensor (or Tensor List) (num relation, rank) (perform inverse outside)
            extra_embs: Dictionary mapping extra (time vec) Strings to extra outputs from extra encoders
        """
        pass
    
    @abstractmethod
    def get_queries(self, queries, ent_embs, rel_embs, extra_embs):
        """Get embedding of queries.

        Args:
            queries: np.array with query triples (head, relation, tail)
            ent_embs: Dictionary mapping encode-ent-content Strings to torch.Tensor (or Tensor List)
            rel_embs: Dictionary mapping encode-rel-content Strings to torch.Tensor (or Tensor List) (num relation, rank) (perform inverse outside)
            extra_embs: Dictionary mapping extra (time vec) Strings to extra outputs from extra encoders
        
        Returns:
            all_anws: torch.Tensor with True query triples' answers (,num query)
            qry_embs: Dictionary mapping encode-query-content Strings to torch.Tensor (or Tensor List) (num query, rank)
            cdd_embs: Dictionary mapping encode-candidate-content Strings to torch.Tensor (or Tensor List)
        """
        pass

    @abstractmethod
    def score(self, qry_embs, cdd_embs):
        """Compute scores of queries against candidates in embedding space.

        Args:
            qry_embs: Dictionary mapping encode-query-content Strings to torch.Tensor (or Tensor List) (num query, rank)
            cdd_embs: Dictionary mapping encode-candidate-content Strings to torch.Tensor (or Tensor List)
            
        Returns:
            scores: torch.Tensor with similarity scores of queries against candidates (num query, num candidate)
        """
        pass

    def forward(self, snapshots, queries, auxiliaries):
        """TKGModel forward pass.

        Args:
            snapshots: List of np.array triples with TKG snapshots
            queries: np.array with query triples (head, relation, tail)
            auxiliaries: Dictionary mapping info Strings to auxiliary information 
            
        Returns:
            ent_embs: Dictionary mapping encode-ent-content Strings to torch.Tensor (or Tensor List)
            rel_embs: Dictionary mapping encode-rel-content Strings to torch.Tensor (or Tensor List) (num relation, rank) (perform inverse outside)
            extra_embs: Dictionary mapping extra (time vec) Strings to extra outputs from extra encoders
            qry_embs: Dictionary mapping encode-query-content Strings to torch.Tensor (or Tensor List) (num query, rank)
            all_anws: torch.Tensor with True query triples' answers (,num query)
            scores: torch.Tensor with query triples' scores (num query, num candidate)
        """
        # get embeddings
        ent_embs, rel_embs, extra_embs = self.get_embeddings(snapshots, auxiliaries)
        # get qry_embs
        all_anws, qry_embs, cdd_embs = self.get_queries(queries, ent_embs, rel_embs, extra_embs)
        # get scores
        scores = self.score(qry_embs, cdd_embs)
        return ent_embs, rel_embs, extra_embs, qry_embs, cdd_embs, all_anws, scores


    def compute_metrics(self, queries, scores, ans4tf):
        """Compute ranking-based evaluation metrics.
    
        Args:
            queries: np.array with query triples (head, relation, tail)
            scores: torch.Tensor with query triples' scores (num query, num candidate)
            ans4tf: Dict with entities for evaluation in the filtered setting

        Returns:
            Evaluation metrics 'raw'/'filter': (mean reciprocical rank, hits@1-3-10, rank)
                mrr: float
                hits@n: List [float, float, float]
                rank: List [int] x n_query
        """
        test_triples = inverse_snapshot(self.sizes[1], queries, self.gpu)
        mrr_filter, mrr_raw, rank_raw, rank_filter = get_total_rank(test_triples, scores, ans4tf)
        hits_raw = get_hits(rank_raw)
        hits_filter = get_hits(rank_filter)

        return {'raw': (mrr_raw, hits_raw, rank_raw),
                'filter': (mrr_filter, hits_filter, rank_filter)}