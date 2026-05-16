import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        h1 = max(64, latent_dim * 4)
        h2 = max(32, latent_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class AAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        hidden_dims = [
            max(512, input_dim),
            max(256, input_dim // 2),
            max(128, input_dim // 4),
            max(64,  input_dim // 8),
        ]
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        self.discriminator = Discriminator(latent_dim)
        self.latent_dim = latent_dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def discriminate(self, z):
        return self.discriminator(z)

    def reconstruct(self, x):
        return self.decoder(self.encoder(x))
