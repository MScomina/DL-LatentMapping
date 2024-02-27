import torch
from torch import nn

import math

class VisualTransformer(nn.Module):
    def __init__(self, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, patch_size=4, expected_dimension=(3, 32, 32)):
        super(VisualTransformer, self).__init__()

        self.patch_size = patch_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.expected_dimension = expected_dimension
        self.latent_space_dim = (expected_dimension[1] // patch_size) * (expected_dimension[2] // patch_size) * expected_dimension[0]
        
        self.transformer = nn.Transformer(
            d_model=(expected_dimension[0] * patch_size * patch_size),
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

    def create_patches(self, x):
        batch_size, channels, height, width = x.shape

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, channels * self.patch_size * self.patch_size)
        patches = patches.to(x.device)

        return patches
    
    def reconstruct_images(self, patches):
        batch_size, _, channels_times_patch_sq = patches.shape
        channels = channels_times_patch_sq // (self.patch_size * self.patch_size)

        # Reshape patches back into small images
        patches = patches.view(batch_size, self.expected_dimension[2] // self.patch_size, self.expected_dimension[1] // self.patch_size, channels, self.patch_size, self.patch_size)

        # Rearrange dimensions
        patches = patches.permute(0, 3, 1, 4, 2, 5)

        # Reshape into original image shape
        images = patches.contiguous().view(batch_size, channels, self.expected_dimension[2], self.expected_dimension[1])

        return images

    
    def positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos = torch.zeros((max_len, d_model))
        pos[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:  # d_model is even
            pos[:, 1::2] = torch.cos(position / div_term)
        else:  # d_model is odd
            pos[:, 1::2] = torch.cos(position / div_term[:-1])  # ignore the last term
        return pos
    
    def encoder(self, x):
        x = self.create_patches(x)
        pos_encoding = self.positional_encoding(x.shape[1], x.shape[2])
        pos_encoding = pos_encoding.unsqueeze(0).to(x.device)
        pos_encoding = pos_encoding.expand(x.shape[0], -1, -1)
        x = x + 0.3*pos_encoding
        x = self.transformer.encoder(x)
        return x
    
    def forward(self, x):
        x = self.create_patches(x)

        pos_encoding = self.positional_encoding(x.shape[1], x.shape[2])
        pos_encoding = pos_encoding.unsqueeze(0).to(x.device)
        pos_encoding = pos_encoding.expand(x.shape[0], -1, -1)
        x = x + 0.3*pos_encoding
        
        x = self.transformer(x, x)
        
        x = self.reconstruct_images(x)
        return x