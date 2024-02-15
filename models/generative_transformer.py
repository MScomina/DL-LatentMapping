import torch
from torch import nn
import numpy as np

import math

class GenerativeImageTransformer(nn.Module):
    def __init__(self, label_dim, noise_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, n_of_classes=10, expected_output_dim=(3, 32, 32)):
        super(GenerativeImageTransformer, self).__init__()

        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.n_of_classes = n_of_classes
        self.expected_output_dim = expected_output_dim

        self.label_embedding = nn.Embedding(n_of_classes, label_dim)
        self.noise_embedding = nn.Linear(noise_dim, noise_dim)

        self.transformer = nn.Transformer(
            d_model=noise_dim + label_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(noise_dim + label_dim)

        self.fc = nn.Linear(noise_dim + label_dim, np.prod(expected_output_dim))

    def forward(self, labels, noise):
        noise = self.noise_embedding(noise) * math.sqrt(self.noise_dim)
        labels = self.label_embedding(labels) * math.sqrt(self.label_dim)
        src = torch.cat((noise, labels), dim=1)
        src = src.unsqueeze(0)
        output = self.transformer(src, src)
        output = self.layer_norm(output)
        output = output.squeeze(0)
        output = self.fc(output)
        output = output.view(-1, *self.expected_output_dim)
        return output