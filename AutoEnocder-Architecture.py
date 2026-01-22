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


# In[ ]:





# In[5]:


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
