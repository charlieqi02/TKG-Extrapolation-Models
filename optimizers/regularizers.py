# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class R0(nn.Module):
    def __init__(self, args):
        super(R0, self).__init__()

    def forward(self, ent_embs, rel_embs, extra_embs, qry_embs, cdd_embs):
        return torch.zeros([1]).cuda()

class Static(nn.Module):
    def __init__(self, args):
        """Static graph constraint, force Angle(evolve emb, static emb) < angle*n.
        
        Args:
            discount: (1 / 0) Ascending pace of the angle {angle, angle*2}(if discount == 1)
            angle: (degree) 
        """
        super(Static, self).__init__()
        self.discount = args.discount
        self.angle = args.angle
        
    def forward(self, ent_embs, rel_embs, extra_embs, qry_embs, cdd_embs):
        evolve_embs = ent_embs['his_embs']
        static_emb = extra_embs['static_emb']
        
        for time_step, evolve_emb in enumerate(evolve_embs):
            step = np.deg2rad(self.angle) * (time_step * self.discount + 1) 
            sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
            mask = (np.cos(step) - sim_matrix) > 0   ## remove ones step >= Angle(evolve emb, static emb)  |  no need for constraint
            loss_static = torch.sum(torch.masked_select(np.cos(step) - sim_matrix, mask))
            
        return loss_static