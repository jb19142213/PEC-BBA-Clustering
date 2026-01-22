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
    max_iter=100,
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


# ------------------------------------------------------------
# Example usage (after Stage 1)
# ------------------------------------------------------------
"""
U = latent features from masked autoencoder (shape: n × d)

result = distributional_evidential_clustering(
    U,
    n_clusters=3,
    beta=2.0,
    gamma=0.05
)

labels = result["labels"]
memberships = result["memberships"]
"""
