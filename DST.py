import numpy as np

# ============================================================
# Stage 3: DST-based Decision Layer (after Stage 2)
# ------------------------------------------------------------
# Input :
#   M  -> fuzzy membership matrix from Stage 2 (n × c)
# Output:
#   final hard labels using DST + pignistic decision
#
# Motivation:
# - Stage 2 models uncertainty at representation/model level
# - DST explicitly models decision-level uncertainty
# - This mirrors classical PEC and improves performance on
#   small, clean datasets like Iris
# ============================================================


# ------------------------------------------------------------
# Convert fuzzy memberships to Basic Belief Assignments (BBA)
# ------------------------------------------------------------
def memberships_to_bba(M):
    """
    Converts fuzzy memberships into DST belief masses.

    For each sample i:
    - mass({j}) = m_ij
    - mass(Ω)   = 1 - max_j m_ij  (ignorance)
    """
    n, c = M.shape
    bbas = []

    for i in range(n):
        masses = {}

        # singleton masses
        for j in range(c):
            masses[frozenset([j])] = M[i, j]

        # ignorance mass
        masses[frozenset(range(c))] = 1.0 - np.max(M[i])

        bbas.append(masses)

    return bbas


# ------------------------------------------------------------
# Dempster's rule of combination (general implementation)
# ------------------------------------------------------------
def dempster_rule(m1, m2, eps=1e-8):
    """
    Combines two BBAs using Dempster's rule.
    """
    combined = {}
    conflict = 0.0

    for A, vA in m1.items():
        for B, vB in m2.items():
            intersection = A & B
            if len(intersection) == 0:
                conflict += vA * vB
            else:
                combined[intersection] = combined.get(intersection, 0.0) + vA * vB

    # normalize
    for key in combined:
        combined[key] /= (1.0 - conflict + eps)

    return combined


# ------------------------------------------------------------
# Pignistic probability decision rule
# ------------------------------------------------------------
def pignistic_decision(bba, n_clusters):
    """
    Converts a BBA into a hard label using pignistic probability.
    """
    betp = np.zeros(n_clusters)

    for A, mass in bba.items():
        for j in A:
            betp[j] += mass / len(A)

    return np.argmax(betp)


# ------------------------------------------------------------
# Full DST post-processing pipeline
# ------------------------------------------------------------
def dst_postprocess(M):
    """
    Applies DST-based decision making to Stage-2 memberships.
    """
    n, c = M.shape
    bbas = memberships_to_bba(M)
    labels = []

    for i in range(n):
        labels.append(pignistic_decision(bbas[i], c))

    return np.array(labels)


# ============================================================
# Example usage (AFTER Stage 2)
# ============================================================
"""
# result = distributional_evidential_clustering(...)
M = result["memberships"]

# DST-enhanced final labels
dst_labels = dst_postprocess(M)

# Evaluate
from sklearn.metrics import adjusted_rand_score
ari_dst = adjusted_rand_score(y_true, dst_labels)
print("ARI with DST:", ari_dst)
"""
