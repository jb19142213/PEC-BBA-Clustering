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


# In[16]:


from sklearn.mixture import GaussianMixture
import numpy as np


# In[17]:


import matplotlib.pyplot as plt
n_components_range = range(1, 20)
bic = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=50)
    gmm.fit(latent_np)
    bic.append(gmm.bic(latent_np))

plt.plot(n_components_range, bic, marker='o')
plt.xlabel("Number of components (n_components)")
plt.ylabel("BIC Score")
plt.title("BIC Score vs. Number of Components")
plt.show()

# The n_components corresponding to the minimum BIC score is often chosen.
optimal_n_components = n_components_range[np.argmin(bic)]
print(f"Optimal n_components based on BIC: {optimal_n_components}")


# In[ ]:





# In[18]:


# Initialize and fit the GMM
gmm = GaussianMixture(n_components=8, covariance_type='full', random_state=42)
gmm.fit(latent_np)

# Predict cluster labels
labels = gmm.predict(latent_np)

# Get probabilities of belonging to each cluster
probabilities = gmm.predict_proba(latent_np)

print("Cluster Labels:\n", labels[:10])
print("\nCluster Probabilities (first 10 samples):\n", probabilities[:10])


# In[ ]:





# In[19]:


cust_data=sdf.select(["n_cust", "timestamp", "cust_c_custseg_lm"]).toPandas()


# In[25]:


cust_data


# In[20]:


cust_data["Label"]=labels


# In[21]:


cust_data["Label"].value_counts()


# In[ ]:





# In[ ]:





# In[23]:


d = dataiku.Dataset("DATA_MANAGEMENT_MY.MAINBANK")
d.read_partitions = ["2025-08"]
mainbank_data= d.get_dataframe()


# In[39]:


df = dataiku.Dataset("OFFUSAUMESTIMATIONMY.PRIORITY_CLUSTERS")
df.read_partitions = ["2025-08"]
cust_old_data= df.get_dataframe()


# In[27]:


cust_data=cust_data.merge(mainbank_data,on=["n_cust", "timestamp", "cust_c_custseg_lm"],how="inner")


# In[29]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# Initialize t-SNE model
# n_components: desired dimensions (e.g., 2 for 2D visualization)
# perplexity: related to the number of nearest neighbors, influences the balance between local and global aspects
tsne = TSNE(n_components=2, random_state=58)


# In[ ]:





# In[30]:


# Fit and transform the data
X_tsne = tsne.fit_transform(X)
latent_tsne=tsne.fit_transform(latent_np)


# In[31]:


X_tsne


# In[41]:


latent_tsne_df=pd.DataFrame({
    "TSNE_1": latent_tsne[:,0],
    "TSNE_2": latent_tsne[:,1]
})


# In[42]:


tsne_df=pd.DataFrame({
    "TSNE_1": X_tsne[:,0],
    "TSNE_2": X_tsne[:,1]
})


# In[43]:


tsne_df["target_label"]=cust_old_data["cluster_label"]


# In[44]:


latent_tsne_df["target_label"]=cust_data["Label"]


# In[ ]:





# In[45]:


plt.figure(figsize=(12, 10))
sns.scatterplot(x='TSNE_1', y='TSNE_2', data=tsne_df, hue='target_label' if 'target_label' in tsne_df.columns else None, palette='viridis')
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()


# In[46]:


plt.figure(figsize=(12, 10))
colors =[
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Yellow-green
    '#17becf',  # Cyan
    '#ff1493',  # Deep Pink
    '#00ff00',  # Lime
    '#ff4500',  # Orange Red
    '#4682b4'   # Steel Blue
]
sns.scatterplot(x='TSNE_1', y='TSNE_2', data=latent_tsne_df, hue='target_label' if 'target_label' in tsne_df.columns else None, palette=colors)
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


print(plt.colormaps())


# In[49]:


def t(x):
    if (x<3):
        return 0
    else:
        return 1
cust_data["hardbank"]=cust_data["mainbank_index"].apply(lambda x : t(x))


# In[51]:


cust_data[["n_cust","timestamp","cust_c_custseg_lm","Label","mainbank_index","hardbank"]]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




