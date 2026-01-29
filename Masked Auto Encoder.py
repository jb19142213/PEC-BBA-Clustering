import pandas as pd
import torch
from torch import nn, optim
import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# =========================
# Masked Variational Autoencoder
# =========================

class MaskedVAE(nn.Module):
    """
    Masked Variational Autoencoder for incomplete data.
    Encoder input: [x ⊙ m , m]
    Latent output: mu (used as latent feature u_i)
    """

    def __init__(self, input_dim, hidden_dim=128, latent_dim=8):
        super().__init__()

        # Encoder: input_dim * 2 because of concatenation with mask
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x, mask):
        x_masked = x * mask
        enc_input = torch.cat([x_masked, mask], dim=1)
        h = self.encoder(enc_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# =========================
# Masked VAE Loss
# =========================

def masked_vae_loss(
    recon_x,
    x,
    mask,
    mu,
    logvar,
    recon_weight=0.7,
    kl_weight=0.3,
    eps=1e-8
):
    """
    Masked reconstruction + KL divergence loss
    """

    # Masked reconstruction loss (MSE over observed entries only)
    se = (recon_x - x) ** 2
    masked_se = se * mask
    recon_loss = masked_se.sum(dim=1) / (mask.sum(dim=1) + eps)
    recon_loss = recon_loss.mean()

    # KL divergence
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = recon_weight * recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss.item(), kl_loss.item()


# =========================
# Training Function
# =========================

def train_masked_vae(
    model,
    dataloader,
    device,
    epochs=50,
    lr=1e-3
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0

        for x, mask in dataloader:
            x = x.to(device).float()
            mask = mask.to(device).float()

            optimizer.zero_grad()
            recon, mu, logvar = model(x, mask)
            loss, rec, kl = masked_vae_loss(recon, x, mask, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rec += rec
            total_kl += kl

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {total_loss:.4f} | "
            f"Recon: {total_rec:.4f} | "
            f"KL: {total_kl:.4f}"
        )

    return model


# =========================
# Latent Feature Extraction
# =========================

def extract_latent_features(model, X, mask, device):
    """
    Returns latent feature vectors u_i = mu_i
    """
    model.eval()
    with torch.no_grad():
        X = X.to(device).float()
        mask = mask.to(device).float()
        mu, _ = model.encode(X, mask)
    return mu.cpu()

