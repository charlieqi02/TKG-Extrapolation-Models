import torch
import torch.nn as nn


class TimeGate(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        
        self.weight = nn.Parameter(torch.Tensor(self.rank, self.rank))   
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        self.bias = nn.Parameter(torch.Tensor(self.rank))
        nn.init.zeros_(self.bias)

        
    def forward(self, embs_input, embs):
        """Update embs using embs_input infomation"""
        time_weight = torch.sigmoid(torch.mm(embs, self.weight) + self.bias)
        embs = time_weight * embs_input + (1-time_weight) * embs
        return embs