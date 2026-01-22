#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pylab', 'inline')


# In[3]:


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


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[29]:


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


# In[6]:


# ============================================
# 1. Load Iris Dataset
# ============================================
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target


# In[7]:


# ============================================
# 2. Inject Missing Values (MCAR)
# ============================================

np.random.seed(42)

n_samples = df.shape[0]
missing_ratio = 0.10
n_missing_rows = int(missing_ratio * n_samples)

missing_rows = np.random.choice(df.index, size=n_missing_rows, replace=False)

for row in missing_rows:
    n_cols_missing = np.random.randint(1, len(iris.feature_names))
    cols_missing = np.random.choice(
        iris.feature_names, size=n_cols_missing, replace=False
    )
    df.loc[row, cols_missing] = np.nan


# In[8]:


# ============================================
# 3. Create Feature Matrix and Mask
# ============================================

X = df[iris.feature_names].values.astype(np.float32)

# Binary mask: 1 = observed, 0 = missing
mask = (~np.isnan(X)).astype(np.float32)

# Fill missing values with zero (mask-aware)
X_filled = np.nan_to_num(X, nan=0.0)


# In[19]:


# ============================================
# 4. Mask-aware Standardization
# ============================================
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

X_scaled = X_filled.copy()
X_tensor = torch.from_numpy(np.array(X_scaled))
scaler = TorchScaler().fit(X_tensor)
X_scaled = scaler.transform(X_tensor)
mask_tensor = torch.tensor(mask, dtype=torch.float32)


# In[10]:



# ============================================
# 5. PyTorch Dataset and DataLoader
# ============================================
    
class MaskedDataset:
    def __init__(self, X, mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx]


dataset = MaskedDataset(X_scaled, mask)
torch.manual_seed(42)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# In[11]:


# =========================
# Example Usage (Skeleton)
# =========================
"""
Assumes your DataLoader yields (x, mask)
x    : tensor of shape [batch_size, input_dim]
mask : tensor of shape [batch_size, input_dim], binary {0,1}
"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MaskedVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
# model = train_masked_vae(model, dataloader, device)
# U = extract_latent_features(model, X_full, mask_full, device)


# In[13]:


import time


# In[31]:


start_time = time.time()

vae = MaskedVAE(input_dim=4, hidden_dim=4, latent_dim=4)
vae = train_masked_vae(vae, dataloader, device,epochs=1000,lr=1e-3)

end_time = time.time()


# In[32]:


elapsed = end_time - start_time
print(f"Code ran for {elapsed:.2f} seconds")


# In[21]:


latent = extract_latent_features(vae, X_scaled,mask_tensor,device)
latent_np = latent.numpy()

