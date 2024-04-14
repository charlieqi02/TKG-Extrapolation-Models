"""Knowledge Graph embedding model optimizer."""
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from utils.train import record_loss, record_metrics


class TKGOptimizer(object):
    """Temporal Knowledge Graph embedding extrapolation model optimizer.

    TKGOptimizers performs loss computations with (negative sampling and) gradient descent steps.

    Attributes:
        model: models.base.TKGModel
        regularizer: regularizers.Regularizer
        optimizer: torch.optim.Optimizer
        reg_alpha: float factor controling the significance of regularization loss
        seq_len: seqence length of input snaps for predicting next snap
        grad_norm: float for clipping gradient (avoid gradient exploding)
    """

    def __init__(self, model, regularizer, optimizer, reg_alpha, label_smoothing, seq_len, grad_norm):
        """Inits TKGOptimizer."""
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=label_smoothing)
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.reg_alpha = reg_alpha
        self.seq_len = seq_len
        self.grad_norm = grad_norm
        self.n_entities = model.sizes[0]
        

    def calculate_loss(self, snapshots, queries, auxiliaries):
        """Compute TKG extrapolation training loss and regularization loss in a batch (for optimization).

        Args:
            snapshots: List of np.array triples with TKG snapshots
            queries: np.array with query triples (head, relation, tail)
            auxiliaries: Dictionary mapping info Strings to auxiliary information 

        Returns:
            loss: torch.Tensor with extrapolation loss and regularization loss
            loss_pst: torch.Tensor with extrapolation loss
            loss_reg: torch.Tensor with regularization loss
            scores: torch.Tensor with query triples' scores (num query, num candidate)
        """
        ent_embs, rel_embs, extra_embs, qry_embs, cdd_embs, all_anws, scores = self.model.forward(
              snapshots, queries, auxiliaries)
        # positive
        loss_pst = self.loss_fn(scores, all_anws)
        # nagative --- have not implemented
        # ...
        # regulatizations
        loss_reg = self.reg_alpha * self.regularizer(ent_embs, rel_embs, extra_embs, qry_embs, cdd_embs)
        
        loss = loss_pst + loss_reg
        return loss, loss_pst, loss_reg, scores


    def get_valid_LossAndMetric(self, his_snaps, valid_snaps, auxiliaries, valid_ans4tf):
        """Compute TKG extrapolation loss and metrics over validdation set (no optimization) Single step inference.

        Args:
            his_snaps: List of np.array containning all snaps' triples before valid_snaps
            valid_snaps: List of np.array containning all validation snaps' triples
            auxiliaries: Dictionary mapping info Strings to auxiliary information
            valid_ans4tf: List of Dictionary of answers for each valid snap

        Returns:
            records: defaultdict(list) each key contains every loss / metric on every snap
                keys: 'loss', 'loss_pst', 'loss_reg', 'mrr_raw', 'hits@1_raw', ..., 'rank_raw' ...
        """
        records = defaultdict(list)
        
        max_snap = len(valid_snaps)
        input_snaps = his_snaps[-self.seq_len:]
        with torch.no_grad():
            with tqdm.tqdm(total=max_snap, unit='snap') as bar:
                bar.set_description(f'valid loss')
                for output_snap, ans4tf in zip(valid_snaps, valid_ans4tf):
                    # compute loss and metrics
                    loss, loss_pst, loss_reg, scores = self.calculate_loss(input_snaps, output_snap, auxiliaries)
                    metrics = self.model.compute_metrics(output_snap, scores, ans4tf)
                    
                    # update input and output
                    input_snaps.pop(0)
                    input_snaps.append(output_snap)
                    
                    # record
                    records = record_loss(records, loss, loss_pst, loss_reg)
                    records = record_metrics(records, metrics)
                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.4f}')
        return records


    def epoch(self, train_snaps, auxiliaries):
        """Runs one epoch of training TKG extrapolation model.

        Args:
            train_snaps: List of np.array containning all training snaps' triples'
            auxiliaries: Dictionary mapping info Strings to auxiliary information 

        Returns:
            records: defaultdict(list) each key contains every loss on every snap
                keys: 'loss', 'loss_pst', 'loss_reg'
        """
        records = defaultdict(list)
        
        max_snap = len(train_snaps)
        time_stamps = np.arange(1, max_snap); random.shuffle(time_stamps)   # exclude snap-0 with no history
        with tqdm.tqdm(total=max_snap-1, unit='snap') as bar:
            bar.set_description(f'train loss')
            for idx in time_stamps:
                # label & input
                output_snap = train_snaps[idx]
                input_snaps = train_snaps[idx-self.seq_len:idx] if idx > self.seq_len else train_snaps[0:idx]

                # update
                loss, loss_pst, loss_reg, scores = self.calculate_loss(input_snaps, output_snap, auxiliaries)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)  # clip gradients
                self.optimizer.step()
                
                # record
                records = record_loss(records, loss, loss_pst, loss_reg)
                bar.update(1)
                bar.set_postfix(loss=f'{loss.item():.4f}')
        return records