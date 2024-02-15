import torch
from torch import nn

class ConditionalAutoEncoder(nn.Module):
    def __init__(self, n_of_classes, latent_space_size, dropout, expected_output_dim = (3, 32, 32)):
        super(ConditionalAutoEncoder, self).__init__()

        self.n_of_classes = n_of_classes
        self.expected_output_dim = expected_output_dim
        self.latent_space_size = latent_space_size
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, latent_space_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_size + n_of_classes, 16 * 7 * 7),
            nn.Softplus(),
            nn.Unflatten(1, (16, 7, 7)),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, label, x):
        x = self.encoder(x)
        x = torch.cat((x, label), dim=1)
        x = self.decoder(x)
        return x