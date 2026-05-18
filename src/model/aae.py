import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        # Shared layers — same structure as before
        shared = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            shared += [
                nn.Linear(prev_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            ]
            prev_dim = h_dim
        self.shared = nn.Sequential(*shared)
        # Two output heads: mean and log-variance of the latent Gaussian
        self.fc_mu     = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers += [
                nn.Linear(prev_dim, h_dim),
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
            max(256, input_dim // 2),
            max(128, input_dim // 4),
            max(64,  input_dim // 8),
        ]
        self.encoder       = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder       = Decoder(latent_dim, hidden_dims, input_dim)
        self.discriminator = Discriminator(latent_dim)
        self.latent_dim    = latent_dim

    def encode(self, x):
        """Deterministic path — returns mu. Used at inference for stable scoring."""
        mu, _ = self.encoder(x)
        return mu

    def encode_stochastic(self, x):
        """Stochastic path — returns (mu, logvar, z) via reparameterisation.
        Used during training so KL gradients flow through the encoder."""
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        z   = mu + std * torch.randn_like(std)
        return mu, logvar, z

    def decode(self, z):
        return self.decoder(z)

    def discriminate(self, z):
        return self.discriminator(z)

    def reconstruct(self, x):
        """Deterministic reconstruction via mu — used for anomaly scoring."""
        return self.decoder(self.encode(x))
