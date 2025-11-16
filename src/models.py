
# src/models.py
import torch
import torch.nn as nn

class SkeletonMLPBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = x.mean(dim=1)
        return self.net(x)

class SkeletonLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        final_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)
