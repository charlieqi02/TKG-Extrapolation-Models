import logging
import os

import torch


class RecoderBestModel(object):
    """Record best model, including record best results, save and load best model"""
    
    def __init__(self, save_dir):
        self.best_model_save_dir = os.path.join(save_dir, "best_models")
        os.makedirs(self.best_model_save_dir, exist_ok=True)
        
        self.best = {'valid': {'raw': (0, -1), 'filter': (0, -1)},
                     'test':  {'raw': (0, -1), 'filter': (0, -1)}}
        self.settings = ['raw', 'filter']
        
    
    def set_best(self, best):
        self.best = best
    
        
    def get_best(self):
        return self.best        
    
            
    def save_best_model(self, model, split, mrrs, epoch):
        """Save best model under {split}_{stt}."""
        for stt in self.settings:
            if self.best[split][stt][0] < mrrs[stt]:
                self.best[split][stt] = (mrrs[stt], epoch)
                save_path = os.path.join(self.best_model_save_dir, f"{split}_{stt}.pt")
                logging.info(f"Saving model as current best at epoch {epoch} - under {split}/{stt}")
                torch.save(model.cpu().state_dict(), save_path)
                model.cuda()


    def load_best_model(self, model, split, setting):
        path = os.path.join(self.best_model_save_dir, f"{split}_{setting}.pt")
        logging.info(f'Loading best mode from {path} ...')
        model.load_state_dict(torch.load(path))
        model.cuda()


    def print_best_models(self, split=None):
        if split is None:
            logging.info("-"*100)
            logging.info("Printing the best models ...")
            for split in self.best.keys():
                logging.info(f"{split} current BEST: ")
                for stt in self.settings:
                    logging.info(f"{stt}\t| mrr {self.best[split][stt][0]:.4f} | at epoch {self.best[split][stt][1]}")
        else:
            logging.info(f"{split} current BEST: ")
            best_info = []
            for stt in self.settings:
                best_info.append(f"mrr_{stt} {self.best[split][stt][0]:.4f} AT {self.best[split][stt][1]}")
            logging.info(best_info[0] + " | " + best_info[1])