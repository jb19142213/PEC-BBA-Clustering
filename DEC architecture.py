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


# In[5]:


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
    recon_weight=1.0,
    kl_weight=0.0,
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


# In[97]:


# ============================================
# 2. Inject Missing Values (MCAR)
# ============================================

np.random.seed(42)

n_samples = df.shape[0]
missing_ratio = 0.0
n_missing_rows = int(missing_ratio * n_samples)

missing_rows = np.random.choice(df.index, size=n_missing_rows, replace=False)

for row in missing_rows:
    n_cols_missing = np.random.randint(1, len(iris.feature_names))
    cols_missing = np.random.choice(
        iris.feature_names, size=n_cols_missing, replace=False
    )
    df.loc[row, cols_missing] = np.nan


# In[98]:


# ============================================
# 3. Create Feature Matrix and Mask
# ============================================

X = df[iris.feature_names].values.astype(np.float32)

# Binary mask: 1 = observed, 0 = missing
mask = (~np.isnan(X)).astype(np.float32)

# Fill missing values with zero (mask-aware)
X_filled = np.nan_to_num(X, nan=0.0)


# In[99]:


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


# In[100]:



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


# In[101]:


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


# In[102]:


import time


# In[103]:


start_time = time.time()

vae = MaskedVAE(input_dim=4, hidden_dim=3, latent_dim=2)
vae = train_masked_vae(vae, dataloader, device,epochs=1000,lr=1e-3)

end_time = time.time()


# In[104]:


elapsed = end_time - start_time
print(f"Code ran for {elapsed:.2f} seconds")


# In[105]:


latent = extract_latent_features(vae, X_scaled,mask_tensor,device)
latent_np = latent.numpy()


# In[106]:


latent_np


# In[ ]:





# In[ ]:


#stage 2


# In[107]:


import numpy as np

# ============================================================
# Stage 2: Distributional Evidential Clustering (Option 3)
# ------------------------------------------------------------
# Input  : U  -> latent features from Stage 1 (masked AE/VAE)
# Output : fuzzy memberships, hard labels, cluster parameters
#
# Key design choices (intentional):
# - Clusters are Gaussian distributions in latent space
# - Memberships are evidential (fuzzy), not probabilistic
# - Distance is negative log-likelihood (not Euclidean)
# - Explicit cluster–cluster divergence regularization
# - Diagonal covariance for stability and efficiency
# - No EM, no mixture weights, no DST
# ============================================================


# ------------------------------------------------------------
# Negative log-likelihood under a diagonal Gaussian
# This replaces point-wise Euclidean distance
# ------------------------------------------------------------
def neg_log_likelihood(u, mu, var, eps=1e-6):
    """
    u   : latent feature vector of a sample (d,)
    mu  : cluster mean (d,)
    var : diagonal variance of cluster (d,)
    """
    d = u.shape[0]
    return 0.5 * (
        np.sum(np.log(var + eps)) +
        np.sum((u - mu) ** 2 / (var + eps)) +
        d * np.log(2 * np.pi)
    )


# ------------------------------------------------------------
# KL divergence between two diagonal Gaussian clusters
# Used to explicitly penalize cluster similarity
# ------------------------------------------------------------
def kl_divergence(mu1, var1, mu2, var2, eps=1e-6):
    """
    Computes KL( N(mu1,var1) || N(mu2,var2) )
    """
    return 0.5 * np.sum(
        np.log((var2 + eps) / (var1 + eps)) +
        (var1 + (mu1 - mu2) ** 2) / (var2 + eps) - 1.0
    )


# ------------------------------------------------------------
# Initialization
# - Means initialized from random latent points
# - Variances initialized to ones
# - Memberships initialized uniformly (maximum uncertainty)
# ------------------------------------------------------------
def initialize_clusters(U, n_clusters, seed=42):
    np.random.seed(seed)
    n, d = U.shape

    idx = np.random.choice(n, n_clusters, replace=False)
    mu = U[idx].copy()
    var = np.ones((n_clusters, d))           # diagonal covariance
    M = np.ones((n, n_clusters)) / n_clusters

    return mu, var, M


# ------------------------------------------------------------
# Compute divergence penalty for each cluster
# Δ_j = sum_{k≠j} KL(C_j || C_k)
# ------------------------------------------------------------
def compute_cluster_penalties(mu, var):
    c = mu.shape[0]
    Delta = np.zeros(c)

    for j in range(c):
        for k in range(c):
            if j != k:
                Delta[j] += kl_divergence(
                    mu[j], var[j],
                    mu[k], var[k]
                )
    return Delta


# ------------------------------------------------------------
# Membership update (Option 3)
# Implements Eq. (4) in the paper
# Uses direct exponentiation (no logsumexp)
# ------------------------------------------------------------
def update_memberships(U, mu, var, beta, gamma, eps=1e-12):
    """
    Memberships are updated using likelihood + divergence cost.
    This is NOT a posterior probability (unlike GMM).
    """
    n, _ = U.shape
    c = mu.shape[0]

    Delta = compute_cluster_penalties(mu, var)
    M = np.zeros((n, c))

    for i in range(n):
        numerators = np.zeros(c)

        for j in range(c):
            # Combined cost:
            # data-to-cluster fit + cluster-separation penalty
            cost_ij = (
                neg_log_likelihood(U[i], mu[j], var[j])
                + gamma * Delta[j]
            )

            numerators[j] = np.exp(-cost_ij / (beta - 1.0))

        # Normalize memberships to sum to 1
        M[i, :] = numerators / (np.sum(numerators) + eps)

    return M


# ------------------------------------------------------------
# Cluster parameter update
# Weighted maximum likelihood using evidential memberships
# Implements Eq. (5) and Eq. (6)
# ------------------------------------------------------------
def update_clusters(U, M, beta, eps=1e-8):
    n, d = U.shape
    c = M.shape[1]

    mu_new = np.zeros((c, d))
    var_new = np.zeros((c, d))
    M_beta = M ** beta

    for j in range(c):
        w = M_beta[:, j][:, None]
        denom = np.sum(w) + eps

        # Mean update
        mu_new[j] = np.sum(w * U, axis=0) / denom

        # Diagonal covariance update
        var_new[j] = np.sum(
            w * (U - mu_new[j]) ** 2, axis=0
        ) / denom

    return mu_new, var_new


# ------------------------------------------------------------
# Main optimization loop (Algorithm 1 in paper)
# ------------------------------------------------------------
def distributional_evidential_clustering(
    U,
    n_clusters,
    beta=2.0,
    gamma=0.05,
    max_iter=3,
    tol=1e-4,
    seed=42
):
    """
    Performs distributional evidential clustering on latent features.
    """
    mu, var, M = initialize_clusters(U, n_clusters, seed)

    for it in range(max_iter):
        mu_old = mu.copy()

        # Step 1: update memberships
        M = update_memberships(U, mu, var, beta, gamma)

        # Step 2: update cluster distributions
        mu, var = update_clusters(U, M, beta)

        # Convergence check on cluster means
        shift = np.max(np.linalg.norm(mu - mu_old, axis=1))
        if shift < tol:
            print(f"Converged at iteration {it}")
            break

    labels = np.argmax(M, axis=1)

    return {
        "memberships": M,
        "labels": labels,
        "mu": mu,
        "var": var
    }


# In[108]:


latent_np.shape


# In[124]:



result = distributional_evidential_clustering(
    latent_np,
    n_clusters=3,
    beta=2,
    gamma=0.00000001,
    max_iter=50,
    tol=1e-8,
    seed=42
)


# In[125]:


result


# In[126]:


labels = result["labels"]
memberships = result["memberships"]
df["pred"]=labels


# In[127]:


from sklearn.metrics import adjusted_rand_score


ari = adjusted_rand_score(df["target"], df["pred"])
print("Adjusted Rand Index:", 100*ari)


# In[93]:


import numpy as np

U = latent_np
n_clusters = 3

mu, var, M = initialize_clusters(U, n_clusters)

print("mu shape:", mu.shape)   # (4, 3)
print("var shape:", var.shape) # (4, 3)
print("M shape:", M.shape)     # (10, 4)


# In[94]:


mu


# In[96]:


var

