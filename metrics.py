"""
metrics.py
Clustering evaluation metrics for algorithms that assign single labels (no meta-clusters)
-1 indicates outliers
"""

import numpy as np
from itertools import combinations
from scipy.special import comb

def Re(original_label, pred_label):
    """
    Error Rate (Re)
    
    Calculates the ratio of misclassifications to total number of objects.
    
    Definition: Re = Ne / N
    where Ne = number of misclassifications
          N  = total number of objects
    
    Misclassification occurs when:
    1. pred_label[i] != original_label[i] (wrong cluster assignment)
    2. pred_label[i] == -1 (outlier) but original_label[i] is not an outlier*
    
    *Note: Assumes original_label has no outlier class (-1)
    
    Parameters:
    -----------
    original_label : array-like, shape (n_samples,)
        Ground truth cluster assignments (non-negative integers)
    pred_label : array-like, shape (n_samples,)
        Predicted cluster assignments (-1 for outliers)
        
    Returns:
    --------
    re : float
        Error rate (0 to 1, lower is better)
    """
    n = len(original_label)
    if n == 0:
        return 0.0
    
    ne = 0  # Number of misclassifications
    
    for i in range(n):
        true_cluster = original_label[i]
        pred_cluster = pred_label[i]
        
        # Misclassification if:
        # 1. Wrong cluster assignment
        # 2. Predicted as outlier (-1) but true is not outlier
        if pred_cluster == -1:
            # If predicted as outlier, always misclassification 
            # (since original_label has no outlier class)
            ne += 1
        elif pred_cluster != true_cluster:
            ne += 1
    
    return ne / n

def Ri(original_label, pred_label):
    """
    Imprecision Rate (Ri)
    
    In your algorithm without meta-clusters, this should always return 0
    since there are no meta-cluster assignments.
    
    Definition: Ri = Ni / N
    where Ni = number of objects in meta-clusters
          N  = total number of objects
    
    Parameters:
    -----------
    original_label : array-like, shape (n_samples,)
        Ground truth cluster assignments
    pred_label : array-like, shape (n_samples,)
        Predicted cluster assignments
        
    Returns:
    --------
    ri : float
        Always 0 (no meta-clusters)
    """
    return 0.0

def EP(original_label, pred_label):
    """
    Evidential Precision (EP) - Simplified for single label assignments
    
    Measures precision of non-outlier assignments.
    
    Definition: EP = TP* / (TP* + FP*)
    where TP* = number of pairs of similar objects (same true cluster) 
                that are both assigned to the same cluster (and not outliers)
          FP* = number of pairs of dissimilar objects (different true clusters)
                that are both assigned to the same cluster (and not outliers)
    
    Only considers pairs where both objects are NOT outliers (pred_label != -1)
    
    Parameters:
    -----------
    original_label : array-like, shape (n_samples,)
        Ground truth cluster assignments
    pred_label : array-like, shape (n_samples,)
        Predicted cluster assignments (-1 for outliers)
        
    Returns:
    --------
    ep : float
        Evidential precision (0 to 1, higher is better)
    """
    n = len(original_label)
    
    # Find indices of non-outlier predictions
    non_outlier_indices = [i for i in range(n) if pred_label[i] != -1]
    
    if len(non_outlier_indices) < 2:
        return 0.0  # Need at least 2 non-outliers to compute pairs
    
    tp_star = 0  # True positive pairs (similar objects in same cluster)
    fp_star = 0  # False positive pairs (dissimilar objects in same cluster)
    
    # Consider all pairs of non-outlier objects
    for i, j in combinations(non_outlier_indices, 2):
        if i >= j:
            continue
            
        true_i = original_label[i]
        true_j = original_label[j]
        pred_i = pred_label[i]
        pred_j = pred_label[j]
        
        # Check if assigned to the same cluster
        if pred_i == pred_j:
            if true_i == true_j:
                tp_star += 1
            else:
                fp_star += 1
    
    denominator = tp_star + fp_star
    return tp_star / denominator if denominator > 0 else 0.0

def ERI(original_label, pred_label):
    """
    Evidential Rank Index (ERI) - Simplified for single label assignments
    
    Comprehensive evaluation considering both beneficial and detrimental elements.
    
    Definition: ERI = 2(TP* + TN*) / [N(N-1)]
    where TP* = number of pairs of similar objects (same true cluster) 
                that are both assigned to the same cluster (and not outliers)
          TN* = number of pairs of dissimilar objects (different true clusters)
                that are assigned to different clusters (and both not outliers)
          N   = total number of objects
    
    Only considers pairs where both objects are NOT outliers (pred_label != -1)
    
    Parameters:
    -----------
    original_label : array-like, shape (n_samples,)
        Ground truth cluster assignments
    pred_label : array-like, shape (n_samples,)
        Predicted cluster assignments (-1 for outliers)
        
    Returns:
    --------
    eri : float
        Evidential rank index (0 to 1, higher is better)
    """
    n = len(original_label)
    
    if n < 2:
        return 0.0
    
    # Find indices of non-outlier predictions
    non_outlier_indices = [i for i in range(n) if pred_label[i] != -1]
    
    tp_star = 0  # True positive pairs
    tn_star = 0  # True negative pairs
    
    # Consider all pairs of non-outlier objects
    for i, j in combinations(non_outlier_indices, 2):
        if i >= j:
            continue
            
        true_i = original_label[i]
        true_j = original_label[j]
        pred_i = pred_label[i]
        pred_j = pred_label[j]
        
        same_true = (true_i == true_j)
        same_pred = (pred_i == pred_j)
        
        if same_true and same_pred:
            tp_star += 1
        elif (not same_true) and (not same_pred):
            tn_star += 1
    
    total_pairs = n * (n - 1)
    return 2 * (tp_star + tn_star) / total_pairs if total_pairs > 0 else 0.0

def rand_index(original_label, pred_label):
    """
    Rand Index (RI) - Standard version
    
    Measures the similarity between two data clusterings.
    
    Definition: RI = (TP + TN) / (TP + FP + FN + TN)
    where TP = number of pairs of similar objects (same true cluster) 
               that are both assigned to the same predicted cluster
          TN = number of pairs of dissimilar objects (different true clusters)
               that are assigned to different predicted clusters
          FP = number of pairs of dissimilar objects 
               that are assigned to the same predicted cluster
          FN = number of pairs of similar objects 
               that are assigned to different predicted clusters
    
    Treats -1 (outliers) as a separate cluster.
    
    Parameters:
    -----------
    original_label : array-like, shape (n_samples,)
        Ground truth cluster assignments
    pred_label : array-like, shape (n_samples,)
        Predicted cluster assignments
        
    Returns:
    --------
    ri : float
        Rand Index (0 to 1, higher is better)
    """
    n = len(original_label)
    
    if n < 2:
        return 0.0
    
    tp = 0  # True positives
    tn = 0  # True negatives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    # Consider all pairs of objects
    for i, j in combinations(range(n), 2):
        if i >= j:
            continue
            
        true_i = original_label[i]
        true_j = original_label[j]
        pred_i = pred_label[i]
        pred_j = pred_label[j]
        
        same_true = (true_i == true_j)
        same_pred = (pred_i == pred_j)
        
        if same_true and same_pred:
            tp += 1
        elif same_true and not same_pred:
            fn += 1
        elif not same_true and same_pred:
            fp += 1
        else:  # not same_true and not same_pred
            tn += 1
    
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0

def adjusted_rand_index(original_label, pred_label):
    """
    Adjusted Rand Index (ARI)
    
    Adjusted for chance version of the Rand Index.
    
    Definition: ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    where RI = Rand Index
          Expected_RI = expected value of RI under random assignment
    
    Parameters:
    -----------
    original_label : array-like, shape (n_samples,)
        Ground truth cluster assignments
    pred_label : array-like, shape (n_samples,)
        Predicted cluster assignments
        
    Returns:
    --------
    ari : float
        Adjusted Rand Index (-1 to 1, higher is better, 0 = random)
    """
    n = len(original_label)
    
    if n < 2:
        return 0.0
    
    # Create contingency table
    true_clusters = np.unique(original_label)
    pred_clusters = np.unique(pred_label)
    
    # Create contingency matrix
    contingency = np.zeros((len(true_clusters), len(pred_clusters)), dtype=int)
    
    # Map clusters to indices
    true_to_idx = {cluster: idx for idx, cluster in enumerate(true_clusters)}
    pred_to_idx = {cluster: idx for idx, cluster in enumerate(pred_clusters)}
    
    # Fill contingency matrix
    for i in range(n):
        true_idx = true_to_idx[original_label[i]]
        pred_idx = pred_to_idx[pred_label[i]]
        contingency[true_idx, pred_idx] += 1
    
    # Calculate row and column sums
    a = contingency.sum(axis=1)  # Row sums
    b = contingency.sum(axis=0)  # Column sums
    
    # Calculate index (sum over cells of n_ij choose 2)
    index = np.sum([comb(n_ij, 2) for n_ij in contingency.flatten() if n_ij >= 2])
    
    # Calculate expected index
    sum_ai = np.sum([comb(ai, 2) for ai in a if ai >= 2])
    sum_bj = np.sum([comb(bj, 2) for bj in b if bj >= 2])
    expected_index = sum_ai * sum_bj / comb(n, 2)
    
    # Calculate max index
    max_index = (sum_ai + sum_bj) / 2
    
    # Calculate ARI
    if max_index == expected_index:
        return 0.0
    else:
        ari = (index - expected_index) / (max_index - expected_index)
        return ari

def compute_all_metrics(original_label, pred_label):
    """
    Compute all 6 clustering metrics.
    
    Parameters:
    -----------
    original_label : array-like, shape (n_samples,)
        Ground truth cluster assignments
    pred_label : array-like, shape (n_samples,)
        Predicted cluster assignments (-1 for outliers)
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Evidential clustering metrics (simplified)
    metrics['Re'] = Re(original_label, pred_label)
    metrics['Ri'] = Ri(original_label, pred_label)  # Always 0
    metrics['EP'] = EP(original_label, pred_label)
    metrics['ERI'] = ERI(original_label, pred_label)
    
    # Standard clustering metrics
    metrics['RI'] = rand_index(original_label, pred_label)
    metrics['ARI'] = adjusted_rand_index(original_label, pred_label)
    
    return metrics

def print_metrics_report(metrics):
    """
    Print a formatted report of all metrics.
    """
    print("=" * 70)
    print("CLUSTERING EVALUATION METRICS")
    print("=" * 70)
    
    print(f"{'Metric':<10} {'Value':<10} {'Description':<50}")
    print("-" * 70)
    
    descriptions = {
        'Re': 'Error rate (misclassification rate, 0-1, lower better)',
        'Ri': 'Imprecision rate (always 0 - no meta-clusters)',
        'EP': 'Evidential Precision (precision on non-outliers, 0-1, higher better)',
        'ERI': 'Evidential Rank Index (0-1, higher better)',
        'RI': 'Rand Index (standard, 0-1, higher better)',
        'ARI': 'Adjusted Rand Index (chance-adjusted, -1 to 1, higher better)'
    }
    
    for metric_name, value in metrics.items():
        desc = descriptions.get(metric_name, '')
        print(f"{metric_name:<10} {value:.4f}     {desc:<50}")
    
    print("=" * 70)

# Additional utility functions
def accuracy(original_label, pred_label, ignore_outliers=False):
    """
    Calculate accuracy of clustering.
    
    Parameters:
    -----------
    original_label : array-like, shape (n_samples,)
        Ground truth cluster assignments
    pred_label : array-like, shape (n_samples,)
        Predicted cluster assignments
    ignore_outliers : bool, default=False
        If True, only consider non-outlier predictions
        
    Returns:
    --------
    acc : float
        Accuracy (0 to 1, higher is better)
    """
    n = len(original_label)
    if n == 0:
        return 0.0
    
    if ignore_outliers:
        # Only consider non-outlier predictions
        mask = pred_label != -1
        if np.sum(mask) == 0:
            return 0.0
        correct = np.sum(original_label[mask] == pred_label[mask])
        return correct / np.sum(mask)
    else:
        # Consider all predictions (outliers always wrong)
        correct = 0
        for i in range(n):
            if pred_label[i] != -1 and pred_label[i] == original_label[i]:
                correct += 1
        return correct / n

def outlier_rate(pred_label):
    """
    Calculate the percentage of points classified as outliers.
    
    Parameters:
    -----------
    pred_label : array-like, shape (n_samples,)
        Predicted cluster assignments
        
    Returns:
    --------
    rate : float
        Percentage of outliers (0 to 1)
    """
    if len(pred_label) == 0:
        return 0.0
    return np.sum(np.array(pred_label) == -1) / len(pred_label)

# Example usage
if __name__ == "__main__":
    # Example 1: Simple case with no outliers
    print("Example 1: No outliers")
    original_label = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred_label = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0])  # Some misclassifications
    
    metrics = compute_all_metrics(original_label, pred_label)
    print_metrics_report(metrics)
    print(f"Accuracy (ignore outliers): {accuracy(original_label, pred_label, ignore_outliers=True):.4f}")
    print()
    
    # Example 2: With outliers
    print("Example 2: With outliers")
    original_label = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred_label = np.array([0, 0, -1, 1, 1, 2, 2, 2, -1])  # Two outliers
    
    metrics = compute_all_metrics(original_label, pred_label)
    print_metrics_report(metrics)
    print(f"Accuracy (ignore outliers): {accuracy(original_label, pred_label, ignore_outliers=True):.4f}")
    print(f"Outlier rate: {outlier_rate(pred_label):.4f}")
    print()
    
    # Example 3: Perfect clustering
    print("Example 3: Perfect clustering")
    original_label = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred_label = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    metrics = compute_all_metrics(original_label, pred_label)
    print_metrics_report(metrics)
