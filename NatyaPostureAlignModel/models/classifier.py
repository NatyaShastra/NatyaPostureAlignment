"""
AdavuClassifier — 3-layer MLP for Bharatanatyam adavu recognition.
~130K parameters. Input: 297-dim feature vector. Output: 14 class logits.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class AdavuClassifier(nn.Module):
    def __init__(self, input_dim: int = 297, num_classes: int = 14,
                 hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdavuDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]