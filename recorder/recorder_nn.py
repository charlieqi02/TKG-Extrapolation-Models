import csv
import logging
import os

import torch
import torch.nn.functional as F


class RecorderInnerModel(object):
    def __init__(self, save_dir, aux_data, train_list, valid_list, test_list):
        self.inner_save_dir = os.path.join(save_dir, "inner_records")
        os.makedirs(self.inner_save_dir, exist_ok=True)
        
        self.atth_paths = [os.path.join(self.inner_save_dir, "atth_entembl2.csv"),
                           os.path.join(self.inner_save_dir, "atth_multic.csv")]
        
        self.aux_data = aux_data
        self.train_list = train_list
        self.valid_list = valid_list
        self.test_list = test_list
        
        
    def atth_record(self, model):
        """Save entities embedding l2 norms as csv files."""
        logging.info("Saving atth ent emb norms and multi-c ...")
        with torch.no_grad():
            entembs = model.ent_emb.weight
            eembl2s = torch.norm(entembs, p=2, dim=1).detach().cpu().numpy()
            multi_c = F.softplus(model.c).squeeze().detach().cpu().numpy()

        with open(self.atth_paths[0], mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(list(eembl2s))
        
        with open(self.atth_paths[1], mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(list(multi_c))