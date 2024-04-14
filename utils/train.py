"""Training utils."""
import datetime
import os
import random

import numpy as np
import torch


def record_loss(records, loss, loss_pst, loss_reg):
    """Update records' loss using given value.
    
    Args:
        records: defaultdict(list)
        loss: torch.Tensor of total loss / batch
        loss_pst: torch.Tensor of extrapolation loss / batch
        loss_reg: torch.Tensor of regularization loss / batch
    Returns:
        records: defaultdict(list) updated
            'loss':
            'loss_pst':
            'loss_reg':
    """
    records['loss'].append(loss.item())
    records['loss_pst'].append(loss_pst.item())
    records['loss_reg'].append(loss_reg.item())
    return records

def record_metrics(records, metrics):
    """Update records' metrics using given value.
    
    Args:
        records: defaultdict(list)
        metrics: Dictionary mapping 'raw'/'filter' to (mrr:float, hits@n:List[float], rank_list:List[int])
    Returns:
        records: defaultdict(list) updated
            'mrr_raw': List containning each snap's mrr under raw setting
            'hits@1_raw': List containning each snap's hits@1 under raw setting
            'hits@3_raw': ...
            'hits@10_raw': ...
            'rank_raw': List containning each snap's rank List for every query under raw setting
            ------ filter setting ------
    """
    settings = ['raw', 'filter']
    hitsN = [1, 3, 10]
    for stt in settings:
        for i, mtc in enumerate(metrics[stt]):
            if   i == 0:
                records[f'mrr_{stt}'].append(mtc)
            elif i == 1:
                for j, N in enumerate(hitsN): 
                    records[f'hits@{N}_{stt}'].append(mtc[j])
            elif i == 2:
                records[f'rank_{stt}'].append(mtc)
    return records


def get_savedir(args):
    """Get unique saving directory name."""
    if not args.continue_train:
        if not args.debug:
            dt = datetime.datetime.now()
            date = dt.strftime("%m_%d")
            save_dir = os.path.join(
                os.environ["LOG_DIR"], date, args.dataset,
                args.model + dt.strftime('_%H_%M_%S')
            )
            os.makedirs(save_dir)
        else:
            save_dir = os.path.join(os.environ["LOG_DIR"], "debug", args.model)
            os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = args.continue_dir
    return save_dir


def count_params(model):
    """Count total number of trainable parameters in model"""
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total


def set_seed(seed):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # PyTorch random number generator.
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python software environment.
    
    if torch.cuda.is_available():  # CUDA random number generator.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # To ensure that CUDA convolution is deterministic.
        torch.backends.cudnn.deterministic = True  
        # If true, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False  

