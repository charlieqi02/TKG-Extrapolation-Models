import logging
import os

import torch


class RecoderCheckpoint(object):
    """Recorde checkpoint, including save and load"""
    
    def __init__(self, save_dir, continue_train, model, optimizer, get_best, set_best):
        self.checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
        self.continue_train = continue_train
        self.get_best = get_best
        self.set_best = set_best
                
        self.start_epoch = 0
        if self.continue_train and os.path.exists(self.checkpoint_path):
            self.start_epoch = self.load_checkpoint(model, optimizer)
           
            
    def save_checkpoint(self, model, optimizer, epoch):
        """Save checkpoint."""
        logging.info(f'Saving checkpoint to {self.checkpoint_path} at epoch {epoch}...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.cpu().state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best': self.get_best()}, self.checkpoint_path)
        model.cuda()


    def load_checkpoint(self, model, optimizer):
        """Load checkpoint.
        Returns:
            start_epoch
        """
        logging.info("Continue training, loading checkpoint ...")
        checkpoint = torch.load(self.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.set_best(checkpoint['best'])
        start_epoch = checkpoint['epoch'] + 1
        model.cuda()
        return start_epoch