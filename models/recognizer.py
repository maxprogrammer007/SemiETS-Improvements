# models/recognizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, in_channels=512, hidden_dim=256, vocab_size=38):
        """
        vocab_size: number of characters + blank
        """
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None))  # (B, 256, 1, W)
        )

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, feat_map):
        """
        feat_map: (B, C, H, W)
        """
        x = self.cnn(feat_map)         # (B, 256, 1, W)
        x = x.squeeze(2)               # (B, 256, W)
        x = x.permute(0, 2, 1)         # (B, W, 256)

        x, _ = self.rnn(x)             # (B, W, 2H)
        logits = self.fc(x)            # (B, W, vocab)

        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
