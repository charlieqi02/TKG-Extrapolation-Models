import torch
import torch.nn as nn


class ConvTransE(torch.nn.Module):
    def __init__(self, rank, channels=50, kernel_size=3, dropout=0):
        super().__init__()
        self.rank = rank
        
        # same size
        padding_size = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(2, channels, kernel_size, stride=1, padding=padding_size)  
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(self.rank)
        self.fc = nn.Linear(self.rank * channels, self.rank)
        self.dropout = nn.Dropout(dropout)


    def forward(self, emb_ent, emb_rel, queries):
        batch_size = len(queries)
        
        cdd_emb = torch.tanh(emb_ent)
        qrye_emb = emb_ent[queries[:, 0]].unsqueeze(1)
        qryr_emb = emb_rel[queries[:, 1]].unsqueeze(1)
        
        x = torch.cat([qrye_emb, qryr_emb], 1)
        x = self.bn0(x)
        x = self.dropout(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.dropout(x)
        
        x = self.bn2(x)
        qry_emb = torch.relu(x)
            
        return qry_emb, cdd_emb