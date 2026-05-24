import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MemoryModule(nn.Module):
    def __init__(self, num_slots, latent_dim, shrink_threshold=0.0025):
        super().__init__()
        self.shrink_threshold = shrink_threshold
        self.memory = nn.Parameter(torch.randn(num_slots, latent_dim))

    def forward(self, z):
        # Cosine similarity between query z and each memory slot
        z_norm = F.normalize(z, dim=1)
        m_norm = F.normalize(self.memory, dim=1)
        attn = F.softmax(torch.matmul(z_norm, m_norm.t()), dim=1)
        # Hard shrinkage — encourages sparse (focused) memory reads
        attn = F.relu(attn - self.shrink_threshold)
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
        z_hat = torch.matmul(attn, self.memory)
        return z_hat, attn


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        h1 = max(64, latent_dim * 4)
        h2 = max(32, latent_dim * 2)
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, h1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(h1, h2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(h2, 1)),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class AAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, num_memory_slots=100):
        super().__init__()
        hidden_dims = [
            max(256, input_dim // 2),
            max(128, input_dim // 4),
            max(64,  input_dim // 8),
        ]
        self.encoder       = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder       = Decoder(latent_dim, hidden_dims, input_dim)
        self.memory        = MemoryModule(num_memory_slots, latent_dim)
        self.discriminator = Discriminator(latent_dim)
        self.latent_dim    = latent_dim

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def read_memory(self, z):
        return self.memory(z)

    def discriminate(self, z):
        return self.discriminator(z)
