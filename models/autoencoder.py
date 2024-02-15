import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, latent_space_size, expected_output_dim = (3, 32, 32), dropout=0.0, patch_size=4, linear_layers=3):
        super(AutoEncoder, self).__init__()

        self.expected_output_dim = expected_output_dim
        self.latent_space_size = latent_space_size
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=expected_output_dim[0], out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(expected_output_dim[1]*expected_output_dim[2], latent_space_size) # 16 * (expected_output_dim[1]*expected_output_dim[2])//16
        )

        self.linear_layers_list = []
        for i in range(linear_layers):
            self.linear_layers_list.append(nn.GELU())
            self.linear_layers_list.append(nn.Dropout(dropout))
            if i != linear_layers-1:
                self.linear_layers_list.append(nn.Linear(latent_space_size, latent_space_size))

        self.linear_layers = nn.Sequential(*self.linear_layers_list)


        self.decoder = nn.Sequential(
            nn.Linear(latent_space_size, expected_output_dim[1]*expected_output_dim[2]), # 16 * (expected_output_dim[1]*expected_output_dim[2])//16
            nn.GELU(),
            nn.Unflatten(1, (16, expected_output_dim[1]//4, expected_output_dim[2]//4)),
            nn.Upsample(scale_factor=round(patch_size/(int)(patch_size**0.5)), mode='nearest'),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=(int)(patch_size**0.5), mode='nearest'),
            nn.Conv2d(in_channels=8, out_channels=expected_output_dim[0], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear_layers(x)
        x = self.decoder(x)
        return x