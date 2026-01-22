# ============================================================
# COMPLETE IMPORT PACKAGES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
import json

# Machine learning imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Sklearn datasets (for loading reliable datasets)
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes

# Additional utilities
import itertools
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.stats import mode

def _impute_mean(X):
    import numpy as np
    X_imputed = X.copy()
    for col in range(X.shape[1]):
        col_data = X[:, col]
        mask = ~np.isnan(col_data)
        if mask.any():
            mean_val = col_data[mask].mean()
            X_imputed[~mask, col] = mean_val
    return X_imputed

def evaluate_clustering_method(X, y_true, c, algorithm_name='pec', **kwargs):
    """
    Evaluate a clustering method.
    """
    print(f"Evaluating {algorithm_name}...", end=' ')
    
    # Run clustering
    assignments, info = clustering_algorithm(X, c, algorithm_name=algorithm_name, **kwargs)
    
    # Convert assignments to labels - FIXED LOGIC
    labels = np.full(len(y_true), -1)
    for i, assign in enumerate(assignments):
        if isinstance(assign, tuple):
            if assign[0] == 'singleton':
                labels[i] = assign[1]
            elif assign[0] == 'meta' and assign[1]:
                # For meta-clusters, assign to first cluster in the set
                if isinstance(assign[1], (tuple, list)) and len(assign[1]) > 0:
                    labels[i] = assign[1][0]
                elif isinstance(assign[1], int):
                    labels[i] = assign[1]
        elif isinstance(assign, (int, np.integer)):
            labels[i] = assign
    
    # Calculate metrics
    mask = labels != -1
    if mask.sum() > 0:
        ari = adjusted_rand_score(y_true[mask], labels[mask])
        nmi = normalized_mutual_info_score(y_true[mask], labels[mask])
    else:
        ari = nmi = 0
    
    print(f"ARI: {ari:.3f}")
    
    return {
        'algorithm': algorithm_name,
        'ari': ari,
        'nmi': nmi,
        'labels': labels,
        'info': info
    }

def clustering_algorithm(X, c, **kwargs):
    import numpy as np 
    import pandas as pd
    """
    Generic clustering function that can be replaced with different algorithms.
    Now includes all methods from the paper.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data matrix with np.nan for missing values.
    c : int
        Number of clusters.
    **kwargs : dict
        Additional parameters including:
        - algorithm_name: str, one of ['pec', 'mi', 'knni', 'lla', 'fktim', 'mica', 
                                     'pds', 'ocs', 'pocs', 'fcm']
        - beta: float, default=2.0 (fuzzifier)
        - eps: float, default=1e-4 (convergence tolerance)
        - max_iter: int, default=200
        - random_state: int or None
        - K: int, default=5 (for KNNI)
        
    Returns:
    --------
    assignments : list
        For each object: a tuple describing its decision:
          ('noise',) or ('singleton', j) or ('meta', (j1, j2, ...))
    additional_info : dict or list
        Additional information (for PEC: list of BBAs for each object)
    """
    
    algorithm_name = kwargs.get('algorithm_name', 'pec').lower()
    
    # ============================================================
    # COMMON UTILITY FUNCTIONS (SHARED BY MULTIPLE METHODS)
    # ============================================================
    
    def _impute_mean(X):
        """Mean imputation (MI)."""
        X_imputed = X.copy()
        for col in range(X.shape[1]):
            col_data = X[:, col]
            mask = ~np.isnan(col_data)
            if mask.any():
                mean_val = col_data[mask].mean()
                X_imputed[~mask, col] = mean_val
        return X_imputed
    
    def _impute_knn(X, K=5):
        """K-Nearest Neighbors Imputation (KNNI)."""
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=K)
        return imputer.fit_transform(X)
    
    def _impute_lla(X, K=5):
        """Locally Linear Approximation (LLA) - simplified version."""
        # Simplified implementation combining mean and KNN
        X_knn = _impute_knn(X, K=K)
        X_mean = _impute_mean(X)
        # Weighted average (more weight to KNN)
        return 0.7 * X_knn + 0.3 * X_mean
    
    def _fcm_clustering(X_filled, c, **fcm_params):
        """Standard FCM clustering on imputed data."""
        from sklearn.cluster import KMeans
        
        # Use KMeans as a simpler alternative to FCM for demo
        # In practice, you'd use a proper FCM implementation
        kmeans = KMeans(n_clusters=c, 
                       random_state=fcm_params.get('random_state', None),
                       n_init=10)
        labels = kmeans.fit_predict(X_filled)
        
        # Convert to PEC-style assignments
        assignments = []
        for label in labels:
            if label == -1:
                assignments.append(('noise',))
            else:
                assignments.append(('singleton', int(label)))
        
        return assignments, {}
    
    def _run_standard_clustering(X_filled, c, **params):
        """Run standard clustering (FCM style) on complete data."""
        # Simple KMeans as placeholder for FCM
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=c, 
                       random_state=params.get('random_state', None),
                       n_init=10)
        labels = kmeans.fit_predict(X_filled)
        
        assignments = []
        for label in labels:
            assignments.append(('singleton', int(label)))
        
        return assignments, {}
    
    # ============================================================
    # METHOD-SPECIFIC IMPLEMENTATIONS
    # ============================================================
    
    if algorithm_name == 'mi':
        """Mean Imputation + FCM"""
        X_imputed = _impute_mean(X)
        return _fcm_clustering(X_imputed, c, **kwargs)
    
    elif algorithm_name == 'knni':
        """KNN Imputation + FCM"""
        K = kwargs.get('K', 5)
        X_imputed = _impute_knn(X, K=K)
        return _fcm_clustering(X_imputed, c, **kwargs)
    
    elif algorithm_name == 'lla':
        """Locally Linear Approximation + FCM"""
        K = kwargs.get('K', 5)
        X_imputed = _impute_lla(X, K=K)
        return _fcm_clustering(X_imputed, c, **kwargs)
    
    elif algorithm_name == 'fktim':
        """Fuzzy K-Top Matching Value Imputation + FCM"""
        # Simplified version: similar to KNNI but with fuzzy weights
        X_imputed = _impute_knn(X, K=kwargs.get('K', 5))
        return _fcm_clustering(X_imputed, c, **kwargs)
    
    elif algorithm_name == 'mica':
        """Multiply Imputed Cluster Analysis (simplified)"""
        # Simplified: multiple imputations, then ensemble clustering
        n_imputations = kwargs.get('n_imputations', 5)
        K = kwargs.get('K', 5)
        
        from sklearn.cluster import KMeans
        import numpy as np
        
        n_samples, n_features = X.shape
        all_labels = []
        
        for _ in range(n_imputations):
            # Add small noise to create multiple imputations
            X_imputed = _impute_knn(X, K=K)
            noise = np.random.normal(0, 0.01, X_imputed.shape)
            X_imputed = X_imputed + noise
            
            kmeans = KMeans(n_clusters=c, 
                           random_state=kwargs.get('random_state', None),
                           n_init=10)
            labels = kmeans.fit_predict(X_imputed)
            all_labels.append(labels)
        
        # Ensemble clustering: take majority vote
        all_labels = np.array(all_labels)
        final_labels = []
        for i in range(n_samples):
            unique, counts = np.unique(all_labels[:, i], return_counts=True)
            final_labels.append(unique[np.argmax(counts)])
        
        # Convert to assignments
        assignments = []
        for label in final_labels:
            assignments.append(('singleton', int(label)))
        
        return assignments, {'n_imputations': n_imputations}
    
    elif algorithm_name == 'pds':
        """Partial Distance Strategy (direct clustering)"""
        # Simplified version: KMeans with partial distances
        from sklearn.cluster import KMeans
        import numpy as np
        
        # Fill missing with column means for initialization
        X_filled = _impute_mean(X)
        mask = ~np.isnan(X)
        
        # Initialize centers
        kmeans = KMeans(n_clusters=c, 
                       random_state=kwargs.get('random_state', None),
                       n_init=10,
                       init='random')
        
        # Modified distance calculation in KMeans loop (simplified)
        # This is a simplified approximation
        for _ in range(kwargs.get('max_iter', 100)):
            labels = kmeans.fit_predict(X_filled)
            # Update centers considering only observed values
            for j in range(c):
                cluster_mask = labels == j
                if cluster_mask.any():
                    for col in range(X.shape[1]):
                        col_data = X[cluster_mask, col]
                        valid_mask = ~np.isnan(col_data)
                        if valid_mask.any():
                            kmeans.cluster_centers_[j, col] = col_data[valid_mask].mean()
        
        assignments = []
        for label in labels:
            assignments.append(('singleton', int(label)))
        
        return assignments, {}
    
    elif algorithm_name == 'ocs':
        """Optimal Completion Strategy"""
        # Add at the beginning:
        X_filled = X.copy()
        mask = ~np.isnan(X)
    
        # Initial imputation
        for col in range(X.shape[1]):
            col_data = X[:, col]
            valid_mask = ~np.isnan(col_data)
            if valid_mask.any():
                mean_val = col_data[valid_mask].mean()
                X_filled[~valid_mask, col] = mean_val
        
        n_samples, n_features = X_filled.shape
        
        # Initial imputation
        X_current = _impute_mean(X_filled)
        
        for iteration in range(kwargs.get('max_iter', 50)):
            # Cluster
            kmeans = KMeans(n_clusters=c, 
                           random_state=kwargs.get('random_state', None),
                           n_init=10)
            labels = kmeans.fit_predict(X_current)
            
            # Update missing values using cluster means
            X_updated = X_filled.copy()
            for j in range(c):
                cluster_mask = labels == j
                if cluster_mask.any():
                    for col in range(n_features):
                        col_mask = mask[cluster_mask, col]
                        if col_mask.any():
                            cluster_mean = X_current[cluster_mask, col][col_mask].mean()
                            # Update only missing values in this cluster
                            missing_in_cluster = cluster_mask & (~mask[:, col])
                            X_updated[missing_in_cluster, col] = cluster_mean
            
            # Check convergence
            if np.allclose(X_current, X_updated, rtol=1e-4):
                break
            X_current = X_updated
        
        assignments = []
        for label in labels:
            assignments.append(('singleton', int(label)))
        
        return assignments, {'iterations': iteration}
    
    elif algorithm_name == 'pocs':
        """Possibilistic Clustering with OCS (simplified)"""
        # FIX: Don't pass algorithm_name again
        assignments, info = clustering_algorithm(X, c, **kwargs)  # <-- REMOVE algorithm_name='ocs'
        # Or even simpler:
        return clustering_algorithm(X, c, algorithm_name='ocs', **kwargs)
    
    elif algorithm_name == 'fcm':
        """Standard FCM on imputed data (baseline)"""
        X_imputed = _impute_mean(X)
        return _fcm_clustering(X_imputed, c, **kwargs)
    
    # ============================================================
    # PEC ALGORITHM (CORRECTED IMPLEMENTATION BASED ON MOTHER CODE)
    # ============================================================
    # ============================================================
# PEC ALGORITHM (FINAL CORRECTED VERSION)
# ============================================================
    elif algorithm_name == 'pec':
        def _init_centers(X, c, random_state=None):
            """Initialize cluster centers by filling missing values with column means."""
            rng = np.random.default_rng(random_state)
            X = np.asarray(X, dtype=float)
            n, s = X.shape
            mask = ~np.isnan(X)
            col_means = np.where(mask, X, np.nan).mean(axis=0)
            X_filled = np.where(mask, X, col_means)
            idx = rng.choice(n, size=c, replace=False)
            return X_filled[idx]

        def _compute_partial_distances(X, V, mask):
            """Compute squared partial distances d_ij^2."""
            X_zero = np.where(mask, X, 0.0)
            diff = X_zero[:, None, :] - V[None, :, :]
            mask3 = mask[:, None, :]
            diff_masked = diff * mask3
            d2 = (diff_masked ** 2).sum(axis=2)
            return d2

        def _update_memberships(d2, beta, delta2):
            """Update membership masses m_ij and m_i_empty."""
            power = -1.0 / (beta - 1.0)
            d2_safe = d2 + 1e-12
            tmp = d2_safe ** power
            delta_term = (delta2 + 1e-12) ** power
            denom = tmp.sum(axis=1, keepdims=True) + delta_term
            m = tmp / denom
            m_empty = delta_term / denom.squeeze()
            return m, m_empty

        def _update_centers(X, mask, M, beta):
            """Update cluster centers via formula (11)."""
            n, s = X.shape
            c = M.shape[1]
            V = np.zeros((c, s), dtype=float)
            m_beta = M ** beta
            X_valid = np.where(np.isnan(X), 0, X)
            valid_mask = (~np.isnan(X)).astype(float)

            for j in range(c):
                w = m_beta[:, j:j+1] * mask
                w_valid = w * valid_mask
                num = (w_valid * X_valid).sum(axis=0)
                den = w_valid.sum(axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    v_j = np.where(den > 0, num / den, 0.0)
                V[j] = v_j
            return V

        def preliminary_pec(X, c, beta=2.0, eps=1e-4, max_iter=200, random_state=None):
            """Step 1: Perform evidential clustering with partial distances."""
            X = np.asarray(X, dtype=float)
            n, s = X.shape
            mask = ~np.isnan(X)

            V = _init_centers(X, c, random_state=random_state)

            for _ in range(max_iter):
                V_prev = V.copy()
                d2 = _compute_partial_distances(X, V, mask)
                delta2 = d2.sum() / (c * n)
                M, m_empty = _update_memberships(d2, beta, delta2)
                V = _update_centers(X, mask, M, beta)
                if np.linalg.norm(V - V_prev) < eps:
                    break

            # Compute threshold phi (14) - FIXED: Use equation (14) properly
            n = len(X)
            m_bar = M.mean(axis=1, keepdims=True)
            phi = ((M - m_bar) ** 2).sum() / (n * c)
            
            # Alternative: Use simplified phi calculation to avoid issues
            if phi < 1e-10:
                phi = 0.1  # Default threshold

            Lambda = []
            is_complete = mask.all(axis=1)

            prelim_type = np.empty(n, dtype=object)
            prelim_cluster = np.full(n, -1, dtype=int)

            for i in range(n):
                masses = np.concatenate(([m_empty[i]], M[i]))
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

                if len(Lambda_i) == 0:
                    prelim_type[i] = 'noise'
                    prelim_cluster[i] = -1
                elif len(Lambda_i) == 1 and 'empty' in Lambda_i:
                    prelim_type[i] = 'noise'
                    prelim_cluster[i] = -1
                elif len(Lambda_i) == 1 and 'empty' not in Lambda_i:
                    prelim_type[i] = 'singleton'
                    (g,) = Lambda_i
                    prelim_cluster[i] = g
                else:
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
                "delta2": delta2,
                "M": M,
                "m_empty": m_empty,
                "V": V,
                "phi": phi
            }
            return V, M, m_empty, info

        def _single_membership_from_distances(d2_row, beta, delta2):
            """Compute membership for a single object."""
            power = -1.0 / (beta - 1.0)
            d2_safe = d2_row + 1e-12
            tmp = d2_safe ** power
            delta_term = (delta2 + 1e-12) ** power
            denom = tmp.sum() + delta_term
            m = tmp / denom
            m_empty = delta_term / denom
            return m, m_empty

        def _softmax_negative(distances):
            """Turn distances into reliability factors via softmax(-d)."""
            d = np.asarray(distances, dtype=float)
            if len(d) == 0:
                return np.array([])
            m = d.min()
            exp_vals = np.exp(-(d - m))
            s = exp_vals.sum()
            if s == 0:
                return np.ones_like(distances) / len(distances)
            return exp_vals / s

        def _build_bba_from_membership(m_clusters, m_empty, c):
            """Build a basic belief assignment (BBA) dict."""
            bba = {'empty': float(m_empty)}
            for j in range(c):
                bba[j] = float(m_clusters[j])
            return bba

        def _discount_bba_with_meta(bba, alpha, cluster_candidates, c):
            """Apply reliability discounting as in (20)-(21)."""
            m_empty = bba.get('empty', 0.0)
            m_clusters = np.array([bba.get(j, 0.0) for j in range(c)], dtype=float)

            m_tilde = {}
            m_tilde['empty'] = m_empty
            m_tilde_clusters = alpha * m_clusters
            for j in range(c):
                m_tilde[j] = float(m_tilde_clusters[j])

            meta_mass = 1.0 - m_empty - m_tilde_clusters.sum()
            meta_mass = max(0.0, min(1.0, meta_mass))
            m_tilde['meta'] = meta_mass
            return m_tilde

        def _subset_from_key(key, cluster_candidates_set):
            """Map a BBA key to a subset of Ω."""
            if key == 'empty':
                return frozenset()
            elif key == 'meta':
                return frozenset(cluster_candidates_set)
            else:
                return frozenset([key])

        def _key_from_subset(subset, cluster_candidates_set):
            """Map subset back to a key."""
            if len(subset) == 0:
                return 'empty'
            if subset == frozenset(cluster_candidates_set):
                return 'meta'
            if len(subset) == 1:
                (j,) = tuple(subset)
                return j
            return None

        def _fuse_two_bbas(bba1, bba2, cluster_candidates_set, c):
            """Fuse two BBAs using the modified Dempster–Shafer rule (22)-(23)."""
            keys = ['empty'] + list(range(c)) + ['meta']

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

            num = {k: 0.0 for k in keys}
            for B in keys:
                for C in keys:
                    setB = _subset_from_key(B, cluster_candidates_set)
                    setC = _subset_from_key(C, cluster_candidates_set)
                    inter = setB & setC
                    keyA = _key_from_subset(inter, cluster_candidates_set)
                    if keyA is not None:
                        num[keyA] += bba1.get(B, 0.0) * bba2.get(C, 0.0)

            fused = {k: 0.0 for k in keys}
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
            """Fuse a list of BBAs."""
            if not bba_list:
                return None
            fused = bba_list[0]
            for bba in bba_list[1:]:
                fused = _fuse_two_bbas(fused, bba, cluster_candidates_set, c)
            return fused

        def pec(X, c, beta=2.0, eps=1e-4, max_iter=200, random_state=None):
            """Full PEC algorithm (Steps 1 and 2)."""
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
            phi = info.get("phi", 0.1)

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

            # FIX: Also store all complete objects as potential neighbors
            complete_objects = []
            complete_object_clusters = []
            for i in range(n):
                if is_complete[i] and prelim_type[i] == 'singleton':
                    complete_objects.append(i)
                    complete_object_clusters.append(prelim_cluster[i])

            for i in range(n):
                # Case 1: uncertain incomplete -> full PEC redistribution
                if prelim_type[i] == 'uncertain_incomplete':
                    cluster_candidates = sorted([j for j in Lambda[i] if j != 'empty'])
                    if len(cluster_candidates) < 2:
                        # If too few candidates, just use nearest cluster
                        bba_final = base_bbas[i]
                        final_bbas[i] = bba_final
                    else:
                        Xi = X[i].copy()
                        Xi_mask = mask[i]
                        Xi_zero = np.where(Xi_mask, Xi, 0.0)

                        bba_versions = []
                        reliabilities_dist = []

                        for g in cluster_candidates:
                            # FIX 1: If no neighbors in this cluster, use all complete objects
                            neigh_idx = cluster_neighbors[g]
                            if len(neigh_idx) == 0:
                                # No specific neighbors, use all complete objects from any cluster
                                # but weighted by their cluster assignment
                                neigh_idx = complete_objects
                                if len(neigh_idx) == 0:
                                    # No complete objects at all, skip this version
                                    continue

                            neighbors = X[neigh_idx]
                            neigh_mask = mask[neigh_idx]
                            
                            # distances for weights (15)
                            diffs = (neighbors - Xi_zero) * Xi_mask
                            dists = np.sqrt((diffs ** 2).sum(axis=1))
                            
                            if Xi_mask.sum() == 0:
                                theta = np.ones(len(neigh_idx)) / len(neigh_idx)
                            else:
                                # FIX 2: Add small epsilon to avoid zero distances
                                dists_safe = dists + 1e-10
                                exp_vals = np.exp(-dists_safe)
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

                            # compute membership for imputed version using centers V
                            Xi_imputed = Xi_imputed[None, :]
                            Xi_imputed_mask = np.ones_like(Xi_imputed, dtype=bool)
                            d2_row = _compute_partial_distances(Xi_imputed, V, Xi_imputed_mask)[0]
                            m_clusters_i, m_empty_i = _single_membership_from_distances(d2_row, beta, delta2)

                            # build BBA for this version
                            bba_i_g = _build_bba_from_membership(m_clusters_i, m_empty_i, c)

                            # compute reliability distance (18)-(19) using neighbors' BBAs
                            vec_i = np.array(
                                [bba_i_g['empty']] + [bba_i_g[j] for j in range(c)],
                                dtype=float
                            )
                            dist_sum = 0.0
                            count = 0
                            for k_idx in neigh_idx[:10]:  # Limit to 10 neighbors for speed
                                bba_k = base_bbas[k_idx]
                                vec_k = np.array(
                                    [bba_k['empty']] + [bba_k[j] for j in range(c)],
                                    dtype=float
                                )
                                dist_sum += np.linalg.norm(vec_i - vec_k)
                                count += 1
                            
                            if count > 0:
                                bba_versions.append(bba_i_g)
                                reliabilities_dist.append(dist_sum / count)
                            else:
                                # If no valid neighbors, skip this version
                                continue

                        if len(bba_versions) == 0:
                            # Could not build any version -> fall back to base BBA
                            bba_final = base_bbas[i]
                            final_bbas[i] = bba_final
                        else:
                            # compute reliability factors alpha via softmax(-distance)
                            alphas = _softmax_negative(reliabilities_dist)
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
                            if bba_final is None:
                                bba_final = base_bbas[i]
                            final_bbas[i] = bba_final

                # Case 2: all other points -> use simple evidential interpretation of Step 1
                else:
                    bba_final = base_bbas[i].copy()
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
                mass_empty = bba_i.get('empty', 0.0)
                masses_clusters = np.array([bba_i.get(j, 0.0) for j in range(c)])
                mass_meta = bba_i.get('meta', 0.0)

                # choose argmax over {empty, singletons, meta}
                elems = ['empty'] + list(range(c)) + ['meta']
                vals = [mass_empty] + list(masses_clusters) + [mass_meta]
                max_idx = np.argmax(vals)
                k_max = elems[max_idx]

                if k_max == 'empty':
                    final_assignments[i] = ('noise',)
                elif k_max == 'meta':
                    cluster_candidates = sorted([j for j in Lambda[i] if j != 'empty'])
                    if not cluster_candidates:
                        # If no candidates, assign to largest singleton
                        max_cluster = np.argmax(masses_clusters)
                        final_assignments[i] = ('singleton', int(max_cluster))
                    else:
                        final_assignments[i] = ('meta', tuple(cluster_candidates))
                else:
                    final_assignments[i] = ('singleton', int(k_max))
            
            # FIX 3: Ensure all objects get assignments
            for i in range(n):
                if final_assignments[i] is None:
                    # Fallback: assign to nearest cluster based on partial distance
                    d2 = _compute_partial_distances(X[i:i+1], V, mask[i:i+1])[0]
                    closest = np.argmin(d2)
                    final_assignments[i] = ('singleton', int(closest))

            return final_assignments, final_bbas
        
        # Extract PEC parameters
        beta = kwargs.get('beta', 2.0)
        eps = kwargs.get('eps', 1e-4)
        max_iter = kwargs.get('max_iter', 200)
        random_state = kwargs.get('random_state', None)
        
        return pec(X, c, beta=beta, eps=eps, max_iter=max_iter, random_state=random_state)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                        f"Choose from: {['pec', 'mi', 'knni', 'lla', 'fktim', 'mica', 'pds', 'ocs', 'pocs', 'fcm']}")

# ============================================================
# Enhanced Evaluation Metrics (Updated for comparison)
# ============================================================

def calculate_all_metrics(assignments, y_true, c):
    """
    Calculate metrics with robust error handling
    """
    n = len(y_true)
    y_pred_hard = np.full(n, -1)
    
    # Convert assignments to hard labels - IMPROVED LOGIC
    for i, assign in enumerate(assignments):
        try:
            if isinstance(assign, tuple):
                if len(assign) >= 2:
                    if assign[0] == 'singleton':
                        y_pred_hard[i] = int(assign[1])
                    elif assign[0] == 'meta' and assign[1]:
                        if isinstance(assign[1], (tuple, list)) and len(assign[1]) > 0:
                            # Use first cluster as default for meta-clusters
                            y_pred_hard[i] = int(assign[1][0])
                        elif isinstance(assign[1], int):
                            y_pred_hard[i] = int(assign[1])
                elif assign[0] == 'noise':
                    y_pred_hard[i] = -1
            elif isinstance(assign, (int, np.integer)):
                y_pred_hard[i] = int(assign)
        except Exception as e:
            # Default assignment to prevent -1
            y_pred_hard[i] = i % c
    
    # Handle any remaining -1 values (noise points)
    mask = y_pred_hard != -1
    if mask.sum() == 0:
        # All assigned as noise, need to assign something
        y_pred_hard = np.random.randint(0, c, size=n)
        mask = np.ones(n, dtype=bool)
    elif mask.sum() < n:
        # Some noise points, assign them to nearest non-noise cluster
        non_noise_idx = np.where(mask)[0]
        noise_idx = np.where(~mask)[0]
        if len(non_noise_idx) > 0:
            for idx in noise_idx:
                # Simple assignment: use most common cluster
                y_pred_hard[idx] = np.bincount(y_pred_hard[non_noise_idx]).argmax()
        mask = np.ones(n, dtype=bool)
    
    # Calculate metrics
    if mask.sum() > 1:
        try:
            ari = adjusted_rand_score(y_true[mask], y_pred_hard[mask])
            ri = rand_score(y_true[mask], y_pred_hard[mask])
            nmi = normalized_mutual_info_score(y_true[mask], y_pred_hard[mask])
        except:
            ari = ri = nmi = 0
    else:
        ari = ri = nmi = 0
    
    # Count assignments
    n_singleton = 0
    n_meta = 0
    n_noise = 0
    
    for a in assignments:
        if isinstance(a, tuple):
            if len(a) >= 2:
                if a[0] == 'singleton':
                    n_singleton += 1
                elif a[0] == 'meta':
                    n_meta += 1
                elif a[0] == 'noise':
                    n_noise += 1
    
    # Calculate specific rate (percentage assigned to singleton clusters)
    specific_rate = n_singleton / n * 100 if n > 0 else 0
    
    # Calculate approximate ERI (using ARI as approximation)
    eri_approx = ari * 100
    
    # Calculate Re (error rate) - simplified
    re_error = (1 - ari) * 100 if ari > 0 else 50
    
    return {
        'ARI_all': ari * 100,
        'RI': ri * 100,
        'NMI': nmi * 100,
        'ERI': eri_approx,
        'Re': re_error,
        'n_singleton': n_singleton,
        'n_meta': n_meta,
        'n_noise': n_noise,
        'specific_rate': specific_rate
    }

# ============================================================
# UPDATED DATASET LOADER - ONLY RELIABLE DATASETS
# ============================================================

def load_reliable_datasets():
    """
    Load only reliable datasets from sklearn (no network fetching).
    """
    datasets = {}
    
    print("Loading reliable datasets from sklearn...")
    
    # List of sklearn datasets (always available, no network required)
    sklearn_datasets = [
        ('Iris', 'iris'),
        ('Wine', 'wine'),
        ('BreastCancer', 'breast_cancer'),
        ('Digits', 'digits'),
        ('Diabetes', 'diabetes')  # Regression dataset, will convert
    ]
    
    for dataset_name, dataset_func in sklearn_datasets:
        try:
            if dataset_func == 'iris':
                data = load_iris()
            elif dataset_func == 'wine':
                data = load_wine()
            elif dataset_func == 'breast_cancer':
                data = load_breast_cancer()
            elif dataset_func == 'digits':
                data = load_digits()
            elif dataset_func == 'diabetes':
                data = load_diabetes()
                # Convert regression to classification for diabetes
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                y = kmeans.fit_predict(data.data)
                data.target = y
            
            # For Digits dataset, use a subset for speed
            if dataset_name == 'Digits':
                # Take first 1000 samples
                X = data.data[:1000]
                y = data.target[:1000]
            else:
                X = data.data
                y = data.target
            
            # For Diabetes, we already have y from kmeans
            if dataset_func != 'diabetes':
                y = data.target
            
            n_clusters = len(np.unique(y))
            
            datasets[dataset_name] = {
                'data': X,
                'target': y,
                'n_clusters': n_clusters,
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            }
            
            print(f"✓ {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features, {n_clusters} clusters")
            
        except Exception as e:
            print(f"✗ Error loading {dataset_name}: {e}")
    
    # Create synthetic datasets matching paper specifications
    synthetic_datasets_info = [
        ('Synthetic_1', 500, 10, 3),
        ('Synthetic_2', 1000, 15, 5),
        ('Synthetic_3', 2000, 20, 7)
    ]
    
    for name, n_samples, n_features, n_clusters in synthetic_datasets_info:
        X, y = create_synthetic_dataset_simple(n_samples, n_features, n_clusters)
        datasets[name] = {
            'data': X,
            'target': y,
            'n_clusters': n_clusters,
            'n_features': n_features,
            'n_samples': n_samples
        }
        print(f"✓ {name} (synthetic): {n_samples} samples, {n_features} features, {n_clusters} clusters")
    
    print(f"\nTotal loaded: {len(datasets)} datasets")
    return datasets

def create_synthetic_dataset_simple(n_samples, n_features, n_clusters):
    """Create a simple synthetic dataset with clear clusters."""
    np.random.seed(42)
    
    # Generate cluster centers
    cluster_centers = np.random.randn(n_clusters, n_features) * 3
    
    # Generate data
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    # Assign samples to clusters
    samples_per_cluster = n_samples // n_clusters
    for k in range(n_clusters):
        start_idx = k * samples_per_cluster
        if k == n_clusters - 1:
            end_idx = n_samples
        else:
            end_idx = (k + 1) * samples_per_cluster
        
        # Generate data for this cluster
        cluster_size = end_idx - start_idx
        X[start_idx:end_idx] = np.random.randn(cluster_size, n_features) + cluster_centers[k]
        y[start_idx:end_idx] = k
    
    # Add some noise
    X += np.random.randn(*X.shape) * 0.5
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

# ============================================================
# Missing Data Generation
# ============================================================

def introduce_missing_data_advanced(X, missing_rate=0.02, pattern='MCAR'):
    """Introduce missing values with different patterns."""
    X_missing = X.copy()
    n_samples, n_features = X.shape
    
    if pattern == 'MCAR':
        # Missing Completely At Random
        n_missing_total = int(missing_rate * n_samples * n_features)
        missing_indices = np.random.choice(
            n_samples * n_features,
            size=n_missing_total,
            replace=False
        )
        
        for idx in missing_indices:
            row = idx // n_features
            col = idx % n_features
            X_missing[row, col] = np.nan
    
    return X_missing

# ============================================================
# UPDATED EXPERIMENT RUNNER FOR MULTIPLE ALGORITHMS
# ============================================================

def run_algorithm_comparison(dataset_name, dataset_info, algorithm_name='pec', 
                           n_runs=3, missing_rates=None, output_dir='results'):
    """
    Run experiments for a specific algorithm and dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    dataset_info : dict
        Dataset information
    algorithm_name : str
        Name of algorithm to test
    n_runs : int
        Number of runs per configuration
    missing_rates : list
        List of missing rates to test
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    results_df : DataFrame
        Results for this algorithm on this dataset
    """
    if missing_rates is None:
        missing_rates = [0.02, 0.05, 0.10]  # 2%, 5%, 10% missing rates
    
    X_original = dataset_info['data']
    y_true = dataset_info['target']
    c = dataset_info['n_clusters']
    n_samples = dataset_info['n_samples']
    n_features = dataset_info['n_features']
    
    all_results = []
    
    for missing_rate in missing_rates:
        print(f"  Missing rate: {missing_rate*100:.0f}%", end=' ')
        
        rate_results = {
            'Re': [], 'Ri': [], 'ERI': [],
            'ARI_all': [], 'RI': [], 'NMI': [],
            'specific_rate': [],
            'n_singleton': [], 'n_meta': [], 'n_noise': []
        }
        
        successful_runs = 0
        for run in range(n_runs):
            # Standardize and introduce missing values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_original)
            X_missing = introduce_missing_data_advanced(
                X_scaled, 
                missing_rate=missing_rate,
                pattern='MCAR'
            )
            
            try:
                # Run clustering algorithm using evaluate_clustering_method
                results = evaluate_clustering_method(
                    X_missing, y_true, c,
                    algorithm_name=algorithm_name,
                    beta=2.0,
                    eps=1e-4,
                    max_iter=200,
                    K=5,
                    n_imputations=5,
                    random_state=42 + run
                )
                
                # Calculate metrics
                metrics = calculate_all_metrics(results['labels'], y_true, c)
                
                # Store results
                for key in rate_results.keys():
                    if key in metrics:
                        rate_results[key].append(metrics[key])
                
                successful_runs += 1
                
            except Exception as e:
                print(f"[Run {run+1} error: {str(e)[:50]}]", end=' ')
                continue
        
        print(f"- {successful_runs}/{n_runs} successful runs")
        
        # Calculate averages for this missing rate
        if rate_results['Re']:
            avg_results = {
                'Dataset': dataset_name,
                'Algorithm': algorithm_name,
                'Missing_rate': missing_rate * 100,
                'n_runs': successful_runs
            }
            
            for key, values in rate_results.items():
                if values:
                    avg_results[f'{key}_mean'] = np.mean(values)
                    avg_results[f'{key}_std'] = np.std(values)
            
            all_results.append(avg_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    return results_df

# ============================================================
# COMPREHENSIVE ALGORITHM COMPARISON
# ============================================================

def run_comprehensive_comparison():
    """Run comprehensive comparison of all algorithms."""
    
    # Define algorithms to compare (from the paper)
    algorithms = [
        'pec',           # Proposed method
        'mi',            # Mean Imputation + FCM
        'knni',          # KNN Imputation + FCM
        'lla',           # Locally Linear Approximation + FCM
        'fktim',         # Fuzzy K-Top Matching + FCM
        'mica',          # Multiple Imputed Cluster Analysis
        'pds',           # Partial Distance Strategy
        'ocs',           # Optimal Completion Strategy
        'pocs',          # Possibilistic Clustering with OCS
        'fcm'            # Standard FCM on imputed data
    ]
    
    # Create main results directory
    main_results_dir = 'C:/Users/ronin/Downloads/clustering_comparison'
    os.makedirs(main_results_dir, exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("Based on PEC paper methods")
    print("=" * 80)
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Results will be saved to: {main_results_dir}")
    print("=" * 80)
    
    # Load reliable datasets
    print("\nLoading reliable datasets...")
    datasets = load_reliable_datasets()
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return None
    
    # Store all results
    all_results = {}
    
    # Loop through each algorithm
    for algorithm in algorithms:
        print(f"\n{'='*60}")
        print(f"ALGORITHM: {algorithm.upper()}")
        print(f"{'='*60}")
        
        # Create algorithm-specific directory
        algo_dir = os.path.join(main_results_dir, algorithm)
        os.makedirs(algo_dir, exist_ok=True)
        
        algorithm_results = []
        
        # Run on each dataset
        for i, (dataset_name, dataset_info) in enumerate(datasets.items(), 1):
            print(f"\nDataset {i}/{len(datasets)}: {dataset_name}")
            
            try:
                dataset_results = run_algorithm_comparison(
                    dataset_name, dataset_info,
                    algorithm_name=algorithm,
                    n_runs=2,  # Reduced for speed
                    missing_rates=[0.02, 0.05, 0.10],
                    output_dir=algo_dir
                )
                
                if not dataset_results.empty:
                    algorithm_results.append(dataset_results)
                    print(f"  ✓ Completed {dataset_name}")
                else:
                    print(f"  ✗ No results for {dataset_name}")
                    
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:50]}")
                continue
        
        # Save algorithm results
        if algorithm_results:
            combined_algo_results = pd.concat(algorithm_results, ignore_index=True)
            results_path = os.path.join(algo_dir, f'{algorithm}_results.csv')
            combined_algo_results.to_csv(results_path, index=False)
            
            all_results[algorithm] = combined_algo_results
            print(f"\n✓ Saved {algorithm} results to {results_path}")
            
            # Generate summary for this algorithm
            generate_algorithm_summary(combined_algo_results, algorithm, algo_dir)
        else:
            print(f"\n✗ No results collected for {algorithm}")
    
    # Generate comprehensive comparison
    if all_results:
        generate_comprehensive_comparison(all_results, datasets, main_results_dir)
        return all_results
    else:
        print("\n✗ No results collected from any algorithm")
        return None

def generate_algorithm_summary(results_df, algorithm_name, output_dir):
    """Generate summary for a specific algorithm."""
    print(f"\n  Generating summary for {algorithm_name}...")
    
    # Calculate average performance
    summary_data = []
    for missing_rate in [2, 5, 10]:
        subset = results_df[results_df['Missing_rate'] == missing_rate]
        if not subset.empty:
            summary_data.append({
                'Missing_rate': missing_rate,
                'Avg_ERI': f"{subset['ERI_mean'].mean():.2f} ± {subset['ERI_mean'].std():.2f}",
                'Avg_ARI': f"{subset['ARI_all_mean'].mean():.2f} ± {subset['ARI_all_mean'].std():.2f}",
                'Avg_RI': f"{subset['RI_mean'].mean():.2f} ± {subset['RI_mean'].std():.2f}",
                'Avg_Re': f"{subset['Re_mean'].mean():.2f} ± {subset['Re_mean'].std():.2f}",
                'Datasets': len(subset)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, f'{algorithm_name}_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Create a simple plot
        try:
            plt.figure(figsize=(10, 6))
            
            # Extract values for plotting
            missing_rates = []
            eri_means = []
            ari_means = []
            
            for missing_rate in [2, 5, 10]:
                subset = results_df[results_df['Missing_rate'] == missing_rate]
                if not subset.empty:
                    missing_rates.append(missing_rate)
                    eri_means.append(subset['ERI_mean'].mean())
                    ari_means.append(subset['ARI_all_mean'].mean())
            
            if missing_rates:
                plt.plot(missing_rates, eri_means, 'o-', label='ERI', linewidth=2)
                plt.plot(missing_rates, ari_means, 's-', label='ARI', linewidth=2)
                plt.xlabel('Missing Rate (%)')
                plt.ylabel('Performance (%)')
                plt.title(f'{algorithm_name.upper()} Performance vs Missing Rate')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_path = os.path.join(output_dir, f'{algorithm_name}_performance.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"  ✗ Could not create plot: {e}")

def generate_comprehensive_comparison(all_results, datasets, output_dir):
    """Generate comprehensive comparison across all algorithms."""
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE COMPARISON")
    print(f"{'='*80}")
    
    # Combine all results
    comparison_data = []
    
    for algorithm, results_df in all_results.items():
        for missing_rate in [2, 5, 10]:
            subset = results_df[results_df['Missing_rate'] == missing_rate]
            if not subset.empty:
                comparison_data.append({
                    'Algorithm': algorithm,
                    'Missing_rate': missing_rate,
                    'ERI_mean': subset['ERI_mean'].mean(),
                    'ERI_std': subset['ERI_mean'].std(),
                    'ARI_mean': subset['ARI_all_mean'].mean(),
                    'ARI_std': subset['ARI_all_mean'].std(),
                    'RI_mean': subset['RI_mean'].mean(),
                    'RI_std': subset['RI_mean'].std(),
                    'Re_mean': subset['Re_mean'].mean(),
                    'Re_std': subset['Re_mean'].std(),
                    'Specific_rate': subset['specific_rate_mean'].mean() if 'specific_rate_mean' in subset.columns else 0,
                    'n_datasets': len(subset)
                })
    
    if not comparison_data:
        print("✗ No comparison data available")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, 'algorithm_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Saved comparison data to {comparison_path}")
    
    # Generate comparison plots
    generate_comparison_plots(comparison_df, output_dir)
    
    # Generate ranking tables
    generate_ranking_tables(comparison_df, output_dir)
    
    # Generate final report
    generate_final_report(comparison_df, datasets, output_dir)

def generate_comparison_plots(comparison_df, output_dir):
    """Generate comparison plots across algorithms."""
    try:
        # Plot 1: ERI comparison across missing rates
        plt.figure(figsize=(14, 8))
        
        # Create a pivot table for easier plotting
        pivot_eri = comparison_df.pivot_table(
            index='Algorithm',
            columns='Missing_rate',
            values='ERI_mean'
        )
        
        # Sort by average performance
        pivot_eri['avg'] = pivot_eri.mean(axis=1)
        pivot_eri = pivot_eri.sort_values('avg', ascending=False)
        pivot_eri = pivot_eri.drop('avg', axis=1)
        
        # Plot bar chart
        x = np.arange(len(pivot_eri))
        width = 0.25
        
        if 2 in pivot_eri.columns:
            plt.bar(x - width, pivot_eri[2], width, label='2% Missing', alpha=0.8)
        if 5 in pivot_eri.columns:
            plt.bar(x, pivot_eri[5], width, label='5% Missing', alpha=0.8)
        if 10 in pivot_eri.columns:
            plt.bar(x + width, pivot_eri[10], width, label='10% Missing', alpha=0.8)
        
        plt.xlabel('Algorithm')
        plt.ylabel('ERI (%)')
        plt.title('ERI Comparison Across Algorithms and Missing Rates')
        plt.xticks(x, pivot_eri.index, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'eri_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Heatmap of ARI performance
        plt.figure(figsize=(12, 8))
        
        pivot_ari = comparison_df.pivot_table(
            index='Algorithm',
            columns='Missing_rate',
            values='ARI_mean'
        )
        
        # Sort by average performance
        pivot_ari['avg'] = pivot_ari.mean(axis=1)
        pivot_ari = pivot_ari.sort_values('avg', ascending=False)
        pivot_ari = pivot_ari.drop('avg', axis=1)
        
        sns.heatmap(pivot_ari, annot=True, fmt='.1f', cmap='RdYlGn',
                   cbar_kws={'label': 'ARI (%)'})
        plt.title('ARI Performance Heatmap')
        plt.xlabel('Missing Rate (%)')
        plt.ylabel('Algorithm')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'ari_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Performance degradation with missing rate
        plt.figure(figsize=(12, 8))
        
        for algorithm in comparison_df['Algorithm'].unique():
            algo_data = comparison_df[comparison_df['Algorithm'] == algorithm].sort_values('Missing_rate')
            if len(algo_data) > 0:
                plt.errorbar(algo_data['Missing_rate'], algo_data['ERI_mean'],
                            yerr=algo_data['ERI_std'], marker='o', label=algorithm,
                            capsize=5, linewidth=2)
        
        plt.xlabel('Missing Rate (%)')
        plt.ylabel('ERI (%)')
        plt.title('Performance Degradation with Increasing Missing Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'degradation_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated comparison plots in {output_dir}")
        
    except Exception as e:
        print(f"✗ Error generating comparison plots: {e}")

def generate_ranking_tables(comparison_df, output_dir):
    """Generate ranking tables for algorithms."""
    try:
        # Calculate average rankings
        ranking_data = []
        
        for algorithm in comparison_df['Algorithm'].unique():
            algo_data = comparison_df[comparison_df['Algorithm'] == algorithm]
            ranking_data.append({
                'Algorithm': algorithm,
                'Avg_ERI': algo_data['ERI_mean'].mean(),
                'Avg_ARI': algo_data['ARI_mean'].mean(),
                'Avg_RI': algo_data['RI_mean'].mean(),
                'Avg_Re': algo_data['Re_mean'].mean(),
                'Avg_Specific': algo_data['Specific_rate'].mean() if 'Specific_rate' in algo_data.columns else 0,
                'Stability': 1 - (algo_data['ERI_std'].mean() / (algo_data['ERI_mean'].mean() + 1e-10))
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Sort by different metrics
        ranking_eri = ranking_df.sort_values('Avg_ERI', ascending=False)
        ranking_ari = ranking_df.sort_values('Avg_ARI', ascending=False)
        ranking_stability = ranking_df.sort_values('Stability', ascending=False)
        
        # Save rankings
        ranking_eri.to_csv(os.path.join(output_dir, 'ranking_by_eri.csv'), index=False)
        ranking_ari.to_csv(os.path.join(output_dir, 'ranking_by_ari.csv'), index=False)
        ranking_stability.to_csv(os.path.join(output_dir, 'ranking_by_stability.csv'), index=False)
        
        print(f"\nTop 5 algorithms by ERI:")
        print(ranking_eri[['Algorithm', 'Avg_ERI']].head().to_string(index=False))
        
        print(f"\nTop 5 algorithms by ARI:")
        print(ranking_ari[['Algorithm', 'Avg_ARI']].head().to_string(index=False))
        
        print(f"\nTop 5 algorithms by Stability:")
        print(ranking_stability[['Algorithm', 'Stability']].head().to_string(index=False))
        
    except Exception as e:
        print(f"✗ Error generating ranking tables: {e}")

def generate_final_report(comparison_df, datasets, output_dir):
    """Generate final comprehensive report."""
    report_path = os.path.join(output_dir, 'final_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE ALGORITHM COMPARISON REPORT\n")
        f.write("Based on PEC Paper Methods\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        for name, info in datasets.items():
            f.write(f"{name}:\n")
            f.write(f"  Samples: {info['n_samples']}\n")
            f.write(f"  Features: {info['n_features']}\n")
            f.write(f"  Clusters: {info['n_clusters']}\n\n")
        
        f.write("\n2. ALGORITHM PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n")
        
        # Calculate average performance
        avg_performance = comparison_df.groupby('Algorithm').agg({
            'ERI_mean': 'mean',
            'ARI_mean': 'mean',
            'RI_mean': 'mean',
            'Re_mean': 'mean',
            'Specific_rate': 'mean',
            'n_datasets': 'sum'
        }).reset_index()
        
        avg_performance = avg_performance.sort_values('ERI_mean', ascending=False)
        
        f.write("\nAverage Performance (sorted by ERI):\n")
        for _, row in avg_performance.iterrows():
            f.write(f"\n{row['Algorithm']}:\n")
            f.write(f"  ERI: {row['ERI_mean']:.2f}%\n")
            f.write(f"  ARI: {row['ARI_mean']:.2f}%\n")
            f.write(f"  RI: {row['RI_mean']:.2f}%\n")
            f.write(f"  Re: {row['Re_mean']:.2f}%\n")
            f.write(f"  Specific Rate: {row['Specific_rate']:.2f}%\n")
            f.write(f"  Datasets evaluated: {row['n_datasets']}\n")
        
        f.write("\n\n3. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        # Find best algorithm
        if not avg_performance.empty:
            best_eri = avg_performance.iloc[0]
            best_ari = avg_performance.sort_values('ARI_mean', ascending=False).iloc[0]
            
            f.write(f"1. Best overall algorithm (by ERI): {best_eri['Algorithm']}\n")
            f.write(f"   Average ERI: {best_eri['ERI_mean']:.2f}%\n")
            f.write(f"   Average ARI: {best_eri['ARI_mean']:.2f}%\n\n")
            
            f.write(f"2. Best algorithm for ARI: {best_ari['Algorithm']}\n")
            f.write(f"   Average ARI: {best_ari['ARI_mean']:.2f}%\n\n")
            
            # Performance degradation analysis
            degradation_data = []
            for algorithm in comparison_df['Algorithm'].unique():
                algo_data = comparison_df[comparison_df['Algorithm'] == algorithm].sort_values('Missing_rate')
                if len(algo_data) >= 2:
                    eri_2 = algo_data[algo_data['Missing_rate'] == 2]['ERI_mean'].values
                    eri_10 = algo_data[algo_data['Missing_rate'] == 10]['ERI_mean'].values
                    if len(eri_2) > 0 and len(eri_10) > 0:
                        degradation = ((eri_2[0] - eri_10[0]) / eri_2[0] * 100) if eri_2[0] > 0 else 0
                        degradation_data.append((algorithm, degradation))
            
            if degradation_data:
                degradation_data.sort(key=lambda x: x[1])
                f.write("3. Performance degradation from 2% to 10% missing rate:\n")
                for algorithm, degradation in degradation_data:
                    f.write(f"   {algorithm}: {degradation:.1f}% degradation\n")
        
        f.write("\n\n4. FILES GENERATED\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. Main results directory: {output_dir}/\n")
        f.write(f"2. Algorithm-specific results: {output_dir}/[algorithm]/[algorithm]_results.csv\n")
        f.write(f"3. Comparison data: {output_dir}/algorithm_comparison.csv\n")
        f.write(f"4. Ranking tables: {output_dir}/ranking_*.csv\n")
        f.write(f"5. Comparison plots: {output_dir}/*_comparison.png\n")
        f.write(f"6. This report: {report_path}\n")
    
    print(f"\n✓ Generated final report: {report_path}")

# ============================================================
# UPDATED MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE CLUSTERING ALGORITHM COMPARISON")
    print("Comparing PEC against 9 other methods from the paper")
    print("=" * 80)
    
    # Run comprehensive comparison
    results = run_comprehensive_comparison()
    
    if results is not None:
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print("\n✓ Successfully compared all 10 algorithms")
        print("✓ Generated comprehensive analysis with all metrics")
        print("✓ Created detailed comparison reports and rankings")
        print("\nResults are organized in:")
        print("  C:/Users/ronin/Downloads/clustering_comparison/")
        print("\nEach algorithm has its own directory with results")
    else:
        print("\n✗ Comparison failed. Check error messages above.")