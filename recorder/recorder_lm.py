import logging
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.test import stat_ranks


class RecorderLossAndMetrics(object):
    """Record loss and metrics, including parse loss/metrics record, console output and tensorboard write"""
    
    def __init__(self, save_dir):
        self.tensorboard_save_dir = os.path.join(save_dir, 'tensorboard')
        self.writer = SummaryWriter(self.tensorboard_save_dir)
        
        self.loss_keys = ["loss", "loss_pst", "loss_reg"]
        self.loss_subs = ["tol", "pst", "reg"]        
        self.settings  = ["raw", "filter"]
        self.hitsN     = [1, 3, 10]
        
        
    def loss_record(self, records, split, epoch):
        """Print and write (in tensorboard) split's loss information.
        
        records keys:
            "loss", "loss_pst", "loss_reg" -> List
        """
        losses = []
        for lkey, sub in zip(self.loss_keys, self.loss_subs):
            losses.append(np.mean(records[lkey]))
            self.writer.add_scalars(f"Loss/{sub}", {split: losses[-1]}, epoch)
            
        logging.info("Epoch {} {}\t| loss: {:.4f} | loss_pst: {:.4f} | loss_reg: {:.4f}".format(
                    epoch, split, losses[0], losses[1], losses[2]))

    def metrics_reocrd(self, records, split, epoch):
        """Print and write (in tensorboard) split's metrics information.
        
        records keys:
            "mrr_{stt}", "hits@{hitsN}_{stt}", "rank_{stt}" -> List
        """       
        mrrs = {}
        for stt in self.settings:
            mrr, hits = stat_ranks(records[f'rank_{stt}'])
            mrrs[stt] = mrr
            logging.info("Epoch {} {}-{}\t| mrr: {:.4f} | hit@1: {:.4f} | hit@3: {:.4f} | hit@10: {:.4f}".format(
                        epoch, split, stt, mrr, hits[0], hits[1], hits[2]))  

            self.writer.add_scalars(f"MRR/{stt}", {split: mrr}, epoch)
            for i, n in enumerate(self.hitsN):
                self.writer.add_scalars(f"Hit{n}/{stt}", {split: hits[i]}, epoch)
            
            # min(max) value / min(max) snap
            self.writer.add_text(f"MRR_min_{stt}/{split}", 
                        f"{np.min(records[f'mrr_{stt}']):.4f} at snap {np.argmin(records[f'mrr_{stt}'])}")
            self.writer.add_text(f"MRR_max_{stt}/{split}", 
                        f"{np.max(records[f'mrr_{stt}']):.4f} at snap {np.argmax(records[f'mrr_{stt}'])}")
        return mrrs

    def close_writer(self):
        self.writer.close()