import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2), nn.Dropout(0.1)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.fc_mu     = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Operates in data space (unlike AAE which operated in latent space).
    Distinguishes real input x from decoder reconstructions."""
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims[:2]:
            layers += [
                nn.utils.spectral_norm(nn.Linear(prev, h)),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ]
            prev = h
        layers += [nn.utils.spectral_norm(nn.Linear(prev, 1)), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VAEGAN(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        # Dynamic hidden dims — same scaling as AAE
        hidden_dims = [
            max(256, input_dim // 2),
            max(128, input_dim // 4),
            max(64,  input_dim // 8),
        ]
        self.encoder       = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder       = Decoder(latent_dim, hidden_dims, input_dim)
        self.discriminator = Discriminator(input_dim, hidden_dims)
        self.latent_dim    = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-7, 7))
        return mu + torch.randn_like(std) * std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def reconstruct(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
