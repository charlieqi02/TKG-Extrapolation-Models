import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.test import stat_ranks


class Recorder:
    """For parsing Loss/Metric logging, TensorBoard writing, Model saving and loading"""
    def __init__(self, save_dir, continue_train, model, optimizer):
        self.save_dir = save_dir
        self.continue_train = continue_train
                         
        # tensorboard writer
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'))
        
        # metrics
        self.settings = ['raw', 'filter']
        self.hisN = [1, 3, 10]
        
        # (Best raw mrr, Best raw epoch), filter ... for validation set
        self.best = {'valid': {'raw': (0, -1), 'filter': (0, -1)},
                     'test':  {'raw': (0, -1), 'filter': (0, -1)}}
        self.best_model_save_dir = os.path.join(self.save_dir, "best_models")
        os.makedirs(self.best_model_save_dir, exist_ok=True)
        
        # checkpoints
        self.checkpoint_file = os.path.join(self.save_dir, 'checkpoint.pt')
        self.start_epoch = 0
        if self.continue_train and os.path.exists(self.checkpoint_file):
            self.start_epoch = self.load_checkpoint(model, optimizer)
        

    def log_train_records(self, records, epoch):
        """Print and write (in tensorboard) train loss information.
        
        Args:
            records: Dictionary mapping ["loss", "loss_pst", "loss_reg"] to List
            epoch: int
        """
        loss = np.mean(records['loss'])
        loss_pst = np.mean(records['loss_pst'])
        loss_reg = np.mean(records['loss_reg'])
        logging.info("Epoch {} train\t| loss: {:.4f} | loss_pst: {:.4f} | loss_reg: {:.4f}".format(
                            epoch, loss, loss_pst, loss_reg))
        self.writer.add_scalars("Loss/tol", {"train": loss}, epoch)
        self.writer.add_scalars("Loss/pst", {"train": loss_pst}, epoch)
        self.writer.add_scalars("Loss/reg", {"train": loss_reg}, epoch)


    def log_valid_records_and_save_best(self, records, epoch, split, model):
        """Print and write (in tensorboard) valid/test loss and metrics information,
            including saving the current one if it's the best so far.
        
        Args:
            records: Dictionary mapping ["loss", "loss_pst", "loss_reg", "mrr_raw", 
                     "hits@1_raw", ..., "rank_raw", ...] to List
            epoch: int
            split: 'valid' / 'test'
            model: nn.Model we are using
        """
        loss = np.mean(records['loss'])
        loss_pst = np.mean(records['loss_pst'])
        loss_reg = np.mean(records['loss_reg'])
        logging.info("Epoch {} {}\t| loss: {:.4f} | loss_pst: {:.4f} | loss_reg: {:.4f}".format(
                            epoch, split, loss, loss_pst, loss_reg))
        self.writer.add_scalars("Loss/tol", {split: loss}, epoch)
        self.writer.add_scalars("Loss/pst", {split: loss_pst}, epoch)
        self.writer.add_scalars("Loss/reg", {split: loss_reg}, epoch)
        
        mrrs = []
        for stt in self.settings:
            logging.info("Epoch {} {}-best-{}(%)\t| mrr: {:.3f} | epoch: {}".format(
                        epoch, split, stt, self.best[split][stt][0]*100, self.best[split][stt][1]))
            
            mrr, hits = stat_ranks(records[f'rank_{stt}'])
            mrrs.append(mrr)
            logging.info("Epoch {} {}-curt-{}(%)\t| mrr: {:.3f} | hit@1: {:.3f} | hit@3: {:.3f} | hit@10: {:.3f}".format(
                            epoch, split, stt, mrr*100, hits[0]*100, hits[1]*100, hits[2]*100))  
            
            if self.best[split][stt][0] < mrr:
                self.best[split][stt] = (mrr, epoch)
                self.save_best_model(model, epoch, split, stt)
            
            self.writer.add_scalars(f"MRR/{stt}", {split: mrr*100}, epoch)
            self.writer.add_scalars(f"Hit1/{stt}", {split: hits[0]*100}, epoch)
            self.writer.add_scalars(f"Hit3/{stt}", {split: hits[1]*100}, epoch)
            self.writer.add_scalars(f"Hit10/{stt}", {split: hits[2]*100}, epoch)
            # min(max) value / min(max) snap
            self.writer.add_text(f"MRR_min_{stt}/{split}", f"{np.min(records[f'mrr_{stt}'])*100},{np.argmin(records[f'mrr_{stt}'])}")
            self.writer.add_text(f"MRR_max_{stt}/{split}", f"{np.max(records[f'mrr_{stt}'])*100},{np.argmax(records[f'mrr_{stt}'])}")


    def save_best_model(self, model, epoch, split, setting):
        """Save best model under valid/test_raw/filter.
        
        Args:
            model: nn.Model
            epoch: int
            split: 'valid' / 'test'
            setting: 'raw' / 'filter'
        """
        logging.info("Saving model at epoch {} in {} under {}/{}".format(
                                epoch, self.best_model_save_dir, split, setting))
        torch.save(model.cpu().state_dict(), 
                   os.path.join(self.best_model_save_dir, f"{split}_{setting}.pt"))
        model.cuda()


    def load_best_model(self, model, split, setting):
        file = os.path.join(self.best_model_save_dir, f"{split}_{setting}.pt")
        logging.info(f'Loading best mode from {file} ...')
        model.load_state_dict(torch.load(file))
        model.cuda()


    def print_best_models(self):
        logging.info("-"*100)
        logging.info("Printing the best models ...")
        for split in self.best.keys():
            logging.info(f"{split}  BEST: ")
            for stt in self.settings:
                logging.info(f"{stt}\t| mrr(%) {self.best[split][stt][0]*100} | at epoch {self.best[split][stt][1]}")


    def save_checkpoint(self, model, optimizer, epoch):
        """Save checkpoint."""
        logging.info(f'Saving checkpoint to {self.checkpoint_file} at epoch {epoch}...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.cpu().state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best': self.best}, self.checkpoint_file)
        model.cuda()


    def load_checkpoint(self, model, optimizer):
        """Load checkpoint.
        Returns:
            start_epoch
        """
        logging.info("Continue training, loading checkpoint ...")
        checkpoint = torch.load(self.checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best = checkpoint['best']
        start_epoch = checkpoint['epoch'] + 1
        model.cuda()
        return start_epoch


    def close_writer(self):
        self.writer.close()