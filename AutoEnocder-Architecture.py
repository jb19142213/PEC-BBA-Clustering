#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import seaborn as sns


# In[2]:


import torch
from torch import nn, optim

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


device


import torch
import torch.nn as nn
import torch.optim as optim

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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.1)
        )

        self.fc_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 4, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
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


# In[6]:


from scmac.spark import start_spark_session
import dataiku.spark as dkuspark

_, sql_context = start_spark_session("test")
d = dataiku.Dataset("DATA_FOR_CLUSTER_DEMO")
d.read_partitions = ["2025-08"]
sdf = dkuspark.get_dataframe(sql_context, d)


# In[7]:


X = sdf.toPandas().drop(["n_cust", "timestamp", "cust_c_custseg_lm"], axis=1)


# In[8]:


X_tensor = torch.from_numpy(np.array(X))
scaler = TorchScaler().fit(X_tensor)
X_scaled = scaler.transform(X_tensor)


# In[9]:


X_scaled


# In[10]:


torch.manual_seed(42)
batch_size = 128
ds = TensorDataset(X_scaled)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


# In[ ]:





# In[11]:


vae = VAE(input_dim=13, hidden_dim=12, latent_dim=3)
vae = train_vae(vae, loader, n_epochs=150, lr=1e-3)


# In[ ]:





# In[12]:


latent = extract_latent(vae, X_scaled)
latent_np = latent.numpy()


# In[13]:


latent_np


# In[14]:


df=pd.DataFrame({
    "F1":latent_np[:,0],
    "F2":latent_np[:,1],
    "F3":latent_np[:,2]
#    "F4":latent_np[:,3]
})


# In[15]:


df.cov()
