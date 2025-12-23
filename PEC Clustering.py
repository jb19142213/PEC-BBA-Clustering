#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Set random seed for reproducibility
np.random.seed(100)

# Number of rows to have missing values
n_rows = df.shape[0]
n_missing = int(0* n_rows)

# Randomly select row indices
missing_rows = np.random.choice(df.index, size=n_missing, replace=False)

# For each selected row, randomly set 1 or more columns (excluding 'target') to NaN
for row in missing_rows:
    n_cols_missing = np.random.randint(1, len(iris.feature_names) + 1)
    cols_missing = np.random.choice(iris.feature_names, size=n_cols_missing, replace=False)
    df.loc[row, cols_missing] = np.nan

def _init_centers(X, c, random_state=None):
    """
    Initialize cluster centers by filling missing values with column means
    and picking random rows as initial centers.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n, s = X.shape
    mask = ~np.isnan(X)
    col_means = np.where(mask, X, np.nan).mean(axis=0)
    X_filled = np.where(mask, X, col_means)
    idx = rng.choice(n, size=c, replace=False)
    return X_filled[idx]





def _compute_partial_distances(X, V, mask):
    """
    Compute squared partial distances d_ij^2 between each object and each center:
      d_ij^2 = sum_p lambda_ip (x_ip - v_jp)^2
    where lambda_ip = 1 if observed, 0 if missing.
    """
    # X_zero has 0 for missing values to avoid nan propagation
    X_zero = np.where(mask, X, 0.0)
    # shape: (n, c, s)
    diff = X_zero[:, None, :] - V[None, :, :]
    # mask3: (n, 1, s) -> broadcast to (n, c, s)
    mask3 = mask[:, None, :]
    diff_masked = diff * mask3
    d2 = (diff_masked ** 2).sum(axis=2)  # (n, c)
    return d2




def _update_memberships(d2, beta, delta2):
    """
    Update membership masses m_ij and m_i_empty (for outliers) given distances d2.
    Uses formulas (10) in the paper with numeric stabilisation.
    """
    power = -1.0 / (beta - 1.0)
    # avoid division by zero
    d2_safe = d2 + 1e-12
    tmp = d2_safe ** power  # shape (n, c)
    delta_term = (delta2 + 1e-12) ** power
    denom = tmp.sum(axis=1, keepdims=True) + delta_term  # shape (n, 1)
    m = tmp / denom  # m_ij
    m_empty = delta_term / denom.squeeze()  # m_i∅
    return m, m_empty




def _update_centers(X, mask, M, beta):
    """
    Update cluster centers via formula (11).
    """
    n, s = X.shape
    c = M.shape[1]
    V = np.zeros((c, s), dtype=float)
    m_beta = M ** beta  # (n, c)

    for j in range(c):
        # weights for cluster j: shape (n, 1)
        w = m_beta[:, j:j+1] * mask  # (n, s)
        num = (w * X).sum(axis=0)    # (s,)
        den = w.sum(axis=0)          # (s,)
        # avoid division by zero: if den[p] == 0, keep center dim as 0
        with np.errstate(divide='ignore', invalid='ignore'):
            v_j = np.where(den > 0, num / den, 0.0)
        V[j] = v_j
    return V




# ============================================================
#  Step 1: Preliminary partial-distance evidential clustering
# ============================================================

def preliminary_pec(X, c, beta=2.0, eps=1e-4, max_iter=200, random_state=None):
    """
    Step 1: Perform evidential clustering with partial distances (no imputation).
    Returns:
      V          : final cluster centers (c, s)
      M          : mass matrix for singleton clusters (n, c)
      m_empty    : mass vector for outlier (n,)
      info       : dict with Lambda sets, completeness, and preliminary labels
    """
    X = np.asarray(X, dtype=float)
    n, s = X.shape
    mask = ~np.isnan(X)  # True if observed

    # 1) Initialize centers
    V = _init_centers(X, c, random_state=random_state)

    # 2) Iterative optimization of objective (5)
    for _ in range(max_iter):
        V_prev = V.copy()
        # distances and delta^2 (7)
        d2 = _compute_partial_distances(X, V, mask)  # (n, c)
        delta2 = d2.sum() / (c * n)
        # memberships (10)
        M, m_empty = _update_memberships(d2, beta, delta2)
        # centers (11)
        print(V_prev)
        V = _update_centers(X, mask, M, beta)
        # check convergence
        if np.linalg.norm(V - V_prev) < eps:
            break

    # 3) Compute threshold phi (14)
    m_bar = M.mean(axis=1, keepdims=True)   # (n,1)
    phi = ((M - m_bar) ** 2).mean()         # scalar

    # 4) Compute Lambda_i sets and preliminary labels
    Lambda = []
    is_complete = mask.all(axis=1)  # (n,)

    prelim_type = np.empty(n, dtype=object)
    prelim_cluster = np.full(n, -1, dtype=int)  # singleton cluster index or -1
    # 'noise' = pure outlier; 'singleton' = definite cluster; 'meta' = meta-cluster; 'uncertain_incomplete' = go to Step 2

    for i in range(n):
        # extended masses: index 0 = empty, 1..c = clusters 0..c-1
        masses = np.concatenate(([m_empty[i]], M[i]))  # shape (c+1,)
        max_idx = np.argmax(masses)
        max_val = masses[max_idx]

        Lambda_i = set()
        for j in range(c + 1):
            if abs(max_val - masses[j]) < phi:
                if j == 0:
                    Lambda_i.add('empty')
                else:
                    Lambda_i.add(j - 1)
        Lambda.append(Lambda_i)

        # classify preliminarily
        if len(Lambda_i) == 0:
            prelim_type[i] = 'noise'
            prelim_cluster[i] = -1
        elif len(Lambda_i) == 1 and 'empty' in Lambda_i:
            prelim_type[i] = 'noise'
            prelim_cluster[i] = -1
        elif len(Lambda_i) == 1 and 'empty' not in Lambda_i:
            # definite singleton cluster
            prelim_type[i] = 'singleton'
            (g,) = Lambda_i
            prelim_cluster[i] = g
        else:
            # ambiguous between >=2 elements
            if is_complete[i]:
                prelim_type[i] = 'meta'
                prelim_cluster[i] = -1
            else:
                prelim_type[i] = 'uncertain_incomplete'
                prelim_cluster[i] = -1

    info = {
        "mask": mask,
        "Lambda": Lambda,
        "is_complete": is_complete,
        "prelim_type": prelim_type,
        "prelim_cluster": prelim_cluster,
        "delta2": delta2
    }
    return V, M, m_empty, info




# ============================================================
#  Step 2: Multiple imputation + DST redistribution
# ============================================================

def _single_membership_from_distances(d2_row, beta, delta2):
    """
    Compute membership (m_j, m_empty) for a single object given distances d2_row (shape (c,)).
    """
    power = -1.0 / (beta - 1.0)
    d2_safe = d2_row + 1e-12
    tmp = d2_safe ** power
    delta_term = (delta2 + 1e-12) ** power
    denom = tmp.sum() + delta_term
    m = tmp / denom
    m_empty = delta_term / denom
    return m, m_empty


def _softmax_negative(distances):
    """
    Turn distances (list/array) into reliability factors via softmax(-d).
    """
    d = np.asarray(distances, dtype=float)
    m = d.min()
    exp_vals = np.exp(-(d - m))  # stabilise
    s = exp_vals.sum()
    if s == 0:
        return np.ones_like(distances) / len(distances)
    return exp_vals / s


def _build_bba_from_membership(m_clusters, m_empty, c):
    """
    Build a basic belief assignment (BBA) dict from membership vector and outlier mass.
    Keys:
      'empty' -> m_empty
      0..c-1  -> m_clusters[j]
    """
    bba = {'empty': float(m_empty)}
    for j in range(c):
        bba[j] = float(m_clusters[j])
    return bba


def _discount_bba_with_meta(bba, alpha, cluster_candidates, c):
    """
    Apply reliability discounting as in (20)-(21), placing uncertainty
    on the meta-cluster corresponding to cluster_candidates.
    Returns a new BBA with keys: 'empty', 0..c-1, 'meta'.
    """
    # original masses
    m_empty = bba.get('empty', 0.0)
    m_clusters = np.array([bba.get(j, 0.0) for j in range(c)], dtype=float)

    m_tilde = {}
    m_tilde['empty'] = m_empty
    # discounted singletons
    m_tilde_clusters = alpha * m_clusters
    for j in range(c):
        m_tilde[j] = float(m_tilde_clusters[j])

    # meta mass
    meta_mass = 1.0 - m_empty - m_tilde_clusters.sum()
    # clip for numerical stability
    meta_mass = max(0.0, min(1.0, meta_mass))
    m_tilde['meta'] = meta_mass
    return m_tilde


def _subset_from_key(key, cluster_candidates_set):
    """
    Map a BBA key to a subset of Ω:
      'empty' -> empty set
      int j   -> {j}
      'meta'  -> cluster_candidates_set
    """
    if key == 'empty':
        return frozenset()
    elif key == 'meta':
        return frozenset(cluster_candidates_set)
    else:
        return frozenset([key])


def _key_from_subset(subset, cluster_candidates_set):
    """
    Map subset back to a key. We only allow:
      ∅       -> 'empty'
      {j}     -> j
      full Λ  -> 'meta'
    """
    if len(subset) == 0:
        return 'empty'
    if subset == frozenset(cluster_candidates_set):
        return 'meta'
    if len(subset) == 1:
        (j,) = tuple(subset)
        return j
    # In this implementation, we don't create new meta-clusters besides Λ;
    # any other subset is ignored (it should not appear in our restricted setup).
    return None


def _fuse_two_bbas(bba1, bba2, cluster_candidates_set, c):
    """
    Fuse two BBAs using the modified Dempster–Shafer rule (22)-(23).
    Keys: 'empty', 0..c-1, 'meta'.
    """
    keys = ['empty'] + list(range(c)) + ['meta']

    # compute conflict K
    K = 0.0
    for B in keys:
        for C in keys:
            if B == 'empty' or C == 'empty':
                continue
            setB = _subset_from_key(B, cluster_candidates_set)
            setC = _subset_from_key(C, cluster_candidates_set)
            inter = setB & setC
            if len(inter) == 0:
                K += bba1.get(B, 0.0) * bba2.get(C, 0.0)

    denom = 1.0 - K + 1e-12

    # fuse non-empty subsets
    num = {k: 0.0 for k in keys}
    for B in keys:
        for C in keys:
            setB = _subset_from_key(B, cluster_candidates_set)
            setC = _subset_from_key(C, cluster_candidates_set)
            inter = setB & setC
            keyA = _key_from_subset(inter, cluster_candidates_set)
            if keyA is None:
                continue
            num[keyA] += bba1.get(B, 0.0) * bba2.get(C, 0.0)

    fused = {k: 0.0 for k in keys}
    # empty case special formula
    fused['empty'] = (
        bba1.get('empty', 0.0)
        + bba2.get('empty', 0.0)
        - bba1.get('empty', 0.0) * bba2.get('empty', 0.0)
    ) / denom

    for k in keys:
        if k == 'empty':
            continue
        fused[k] = num[k] / denom

    return fused


def _fuse_bba_list(bba_list, cluster_candidates_set, c):
    """
    Fuse a list of BBAs with the above fusion rule.
    """
    if not bba_list:
        return None
    fused = bba_list[0]
    for bba in bba_list[1:]:
        fused = _fuse_two_bbas(fused, bba, cluster_candidates_set, c)
    return fused




def pec(X, c, beta=2.0, eps=1e-4, max_iter=200, random_state=None):
    """
    Full PEC algorithm (Steps 1 and 2) as described in the paper:
      - Partial-distance evidential clustering (no imputation)
      - Multiple imputation for uncertain incomplete objects
      - DST-based reliability discounting and evidence fusion

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix with np.nan for missing values.
    c : int
        Number of specific clusters.
    beta : float, optional
        Fuzzifier (usually 2).
    eps : float, optional
        Convergence tolerance for centers.
    max_iter : int, optional
        Maximum number of iterations in Step 1.
    random_state : int or None

    Returns
    -------
    final_assignments : list
        For each object i: a tuple describing its decision:
          ('noise',)
          ('singleton', j)
          ('meta', (j1, j2, ...))
    final_bbas : list of dict
        For each object, its final BBA over:
          'empty', 0..c-1, optionally 'meta'
    """
    X = np.asarray(X, dtype=float)
    n, s = X.shape

    # Step 1: preliminary evidential clustering via partial distances
    V, M, m_empty, info = preliminary_pec(
        X, c, beta=beta, eps=eps, max_iter=max_iter, random_state=random_state
    )
    mask = info["mask"]
    Lambda = info["Lambda"]
    is_complete = info["is_complete"]
    prelim_type = info["prelim_type"]
    prelim_cluster = info["prelim_cluster"]
    delta2 = info["delta2"]

    # Build neighbor pools: complete objects firmly assigned to singleton clusters
    cluster_neighbors = {g: [] for g in range(c)}
    for i in range(n):
        if is_complete[i] and prelim_type[i] == 'singleton':
            g = prelim_cluster[i]
            cluster_neighbors[g].append(i)

    # Step 2: handle uncertain incomplete objects via multiple imputation + DST
    final_bbas = [None] * n
    final_assignments = [None] * n

    # Precompute once: base BBAs from Step 1 for all objects
    base_bbas = []
    for i in range(n):
        bba = _build_bba_from_membership(M[i], m_empty[i], c)
        base_bbas.append(bba)

    for i in range(n):
        # Case 1: uncertain incomplete -> full PEC redistribution
        if prelim_type[i] == 'uncertain_incomplete':
            # candidate clusters (ignore 'empty')
            cluster_candidates = sorted([j for j in Lambda[i] if j != 'empty'])
            if len(cluster_candidates) < 2:
                # not enough info -> fall back to base BBA
                bba_final = base_bbas[i]
                final_bbas[i] = bba_final
            else:
                # multiple imputation: one version per candidate cluster
                Xi = X[i].copy()
                Xi_mask = mask[i]
                Xi_zero = np.where(Xi_mask, Xi, 0.0)

                bba_versions = []
                reliabilities_dist = []

                for g in cluster_candidates:
                    neigh_idx = cluster_neighbors[g]
                    if len(neigh_idx) == 0:
                        continue  # cannot impute from this cluster

                    neighbors = X[neigh_idx]  # (K, s)
                    neigh_mask = mask[neigh_idx]
                    # distances for weights (15)
                    # use Xi's observed dims only
                    diffs = (neighbors - Xi_zero) * Xi_mask  # broadcast Xi_mask (s,)
                    dists = np.sqrt((diffs ** 2).sum(axis=1))  # (K,)
                    # if Xi has no observed dims, give equal weights
                    if Xi_mask.sum() == 0:
                        theta = np.ones(len(neigh_idx)) / len(neigh_idx)
                    else:
                        # weights (16)
                        exp_vals = np.exp(-dists)
                        s_exp = exp_vals.sum()
                        if s_exp == 0:
                            theta = np.ones(len(neigh_idx)) / len(neigh_idx)
                        else:
                            theta = exp_vals / s_exp

                    # impute missing attributes using (17)
                    Xi_imputed = Xi.copy()
                    missing_dims = ~Xi_mask
                    if missing_dims.any():
                        Xi_imputed[missing_dims] = (theta[:, None] * neighbors[:, missing_dims]).sum(axis=0)

                    # compute membership for imputed version using centers V (one step of (10))
                    Xi_imputed = Xi_imputed[None, :]  # shape (1, s)
                    Xi_imputed_mask = np.ones_like(Xi_imputed, dtype=bool)
                    d2_row = _compute_partial_distances(Xi_imputed, V, Xi_imputed_mask)[0]
                    m_clusters_i, m_empty_i = _single_membership_from_distances(d2_row, beta, delta2)

                    # build BBA for this version
                    bba_i_g = _build_bba_from_membership(m_clusters_i, m_empty_i, c)

                    # compute reliability distance (18)-(19) using neighbors' BBAs
                    # here we measure in the space of {empty} + singletons
                    vec_i = np.array(
                        [bba_i_g['empty']] + [bba_i_g[j] for j in range(c)],
                        dtype=float
                    )
                    dist_sum = 0.0
                    for k_idx in neigh_idx:
                        bba_k = base_bbas[k_idx]
                        vec_k = np.array(
                            [bba_k['empty']] + [bba_k[j] for j in range(c)],
                            dtype=float
                        )
                        dist_sum += np.linalg.norm(vec_i - vec_k)
                    bba_versions.append(bba_i_g)
                    reliabilities_dist.append(dist_sum)

                if len(bba_versions) == 0:
                    # could not build any version -> fall back
                    bba_final = base_bbas[i]
                    final_bbas[i] = bba_final
                else:
                    # compute reliability factors alpha via softmax(-distance)
                    alphas = _softmax_negative(reliabilities_dist)
                    # discount each BBA and put ignorance on meta-cluster Λ̂_i
                    cluster_candidates_set = set(cluster_candidates)
                    discounted_list = []
                    for bba_v, alpha in zip(bba_versions, alphas):
                        discounted = _discount_bba_with_meta(
                            bba_v, alpha, cluster_candidates, c
                        )
                        discounted_list.append(discounted)

                    # fuse them using modified DS rule
                    bba_final = _fuse_bba_list(
                        discounted_list, cluster_candidates_set, c
                    )
                    final_bbas[i] = bba_final

        # Case 2: all other points -> use simple evidential interpretation of Step 1
        else:
            bba_final = base_bbas[i].copy()
            # If complete and ambiguous (meta), push mass of ambiguous clusters to 'meta'
            if prelim_type[i] == 'meta':
                cluster_candidates = sorted([j for j in Lambda[i] if j != 'empty'])
                cluster_candidates_set = set(cluster_candidates)
                if cluster_candidates:
                    # collect mass for meta from those candidates
                    meta_mass = sum(bba_final.get(j, 0.0) for j in cluster_candidates)
                    for j in cluster_candidates:
                        bba_final[j] = 0.0
                    bba_final['meta'] = bba_final.get('meta', 0.0) + meta_mass
            final_bbas[i] = bba_final

        # now derive final assignment from final_bbas[i]
        bba_i = final_bbas[i]
        # ensure keys exist
        mass_empty = bba_i.get('empty', 0.0)
        masses_clusters = np.array([bba_i.get(j, 0.0) for j in range(c)])
        mass_meta = bba_i.get('meta', 0.0)

        # choose argmax over {empty, singletons, meta}
        elems = ['empty'] + list(range(c)) + ['meta']
        vals = [mass_empty] + list(masses_clusters) + [mass_meta]
        k_max = elems[int(np.argmax(vals))]

        if k_max == 'empty':
            final_assignments[i] = ('noise',)
        elif k_max == 'meta':
            # for meta we can retrieve candidates from Lambda[i] (excluding 'empty')
            cluster_candidates = sorted([j for j in Lambda[i] if j != 'empty'])
            final_assignments[i] = ('meta', tuple(cluster_candidates))
        else:
            final_assignments[i] = ('singleton', int(k_max))

    return final_assignments, final_bbas


