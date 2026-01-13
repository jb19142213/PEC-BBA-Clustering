import pandas as pd
import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    
    def __init__(self, input_dim=13, hidden_dim = 128, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(hidden_dim, hidden_dim-2),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(hidden_dim-2, hidden_dim-4),
        nn.LeakyReLU(negative_slope=0.1)
        )
        
        self.fc_mu = nn.Linear(hidden_dim-4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim-4, latent_dim)
        
        self.decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim-4),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(hidden_dim-4, hidden_dim-2),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(hidden_dim-2, hidden_dim),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(hidden_dim, input_dim))
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    
def vae_loss(recon_x,x,mu,logvar,mask,recon_weight=0.6,kl_weight=0.4,eps=1e-8):
    """
    Masked VAE loss for missing data.

    recon_x : reconstructed output
    x       : original input
    mu      : latent mean
    logvar  : latent log-variance
    mask    : binary mask (1 = observed, 0 = missing)
    """

    # -------------------------
    # Masked reconstruction loss
    # -------------------------
    squared_error = (recon_x - x) ** 2
    masked_se = squared_error * mask

    # Normalize per sample to avoid bias toward samples with more observed values
    recon_loss = masked_se.sum(dim=1) / (mask.sum(dim=1) + eps)
    recon_loss = recon_loss.mean()

    # -------------------------
    # KL divergence term
    # -------------------------
    kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    # -------------------------
    # Total loss
    # -------------------------
    loss = recon_weight * recon_loss + kl_weight * kl_div

    return loss, recon_loss.item(), kl_div.item()

def vae_loss(recon_x, x, mu, logvar,recon_weight=0.6, kl_weight=0.4):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    #binary_cross_entropy(x_hat, x, reduction='sum')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_weight * recon_loss + kl_weight * kl_div #
    return loss,recon_loss.item(), kl_div.item() #


class TorchScaler:
    def fit(self, X: torch.Tensor):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, unbiased=False, keepdim=True)
        self.std[self.std == 0] = 1.0
        return self
    def transform(self, X: torch.Tensor):
        return (X - self.mean) / self.std
    def inverse_transform(self, X: torch.Tensor):
        return X * self.std + self.mean
    
def train_vae(model, dataloader, n_epochs=20, lr=1e-3, weight_decay=0.0, verbose=True):
    model = model.to(device)
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer= optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-08)
    model.train()
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        batches = 0
        for (xb, ) in dataloader:
            xb = xb.to(device).float()
            optimizer.zero_grad()
            recon, mu, log_var = model(xb)
            loss,recon_l, kl_l = vae_loss(recon, xb, mu, log_var)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_recon += recon_l
            running_kl += kl_l
            batches += 1
        if verbose :
            print(f"Epoch {epoch}/{n_epochs} avg_loss = {running_loss/batches:.6f} recon = {running_recon/batches:.6f} kl={running_kl/batches:.6f}")
    return model

def extract_latent(model, X_tensor):
    model.eval()
    with torch.no_grad():
        X_device = X_tensor.to(device).float()
        mu, logvar = model.encode(X_device)
        return mu.cpu()

