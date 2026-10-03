"""Arabic dialectology: cluster labels and dialectal-distance matrices.

Sources used to derive the distances:
  - Versteegh, K. (2014). "The Arabic Language". Edinburgh Univ. Press.
  - Habash, N. Y. (2006). "Introduction to Arabic Natural Language Processing".
  - Al-Badrashiny et al. (2024). NADI 2024 shared task paper (5-way group labels).

Distance scale: 0.0 = identical dialect; 0.2 = very close (intra-cluster);
0.4 = close (intra macro-region); 0.6 = moderate cross-region;
0.8 = far cross-region (e.g. Khaleeji-Maghrebi).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# NADI 18: 18 country dialects
# Indices: 0=OM, 1=SD, 2=SA, 3=KW, 4=QA, 5=LB, 6=JO, 7=SY,
#          8=IQ, 9=MA, 10=EG, 11=PL, 12=YE, 13=BH, 14=DZ, 15=AE, 16=TN, 17=LY
NADI_18_CLUSTER = {
    0: 0, 1: 3, 2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 2, 8: 1,
    9: 4, 10: 3, 11: 2, 12: 0, 13: 0, 14: 4, 15: 0, 16: 4, 17: 4,
}

KHALEEJI_INTRA = {
    frozenset({2, 3}): 0.05, frozenset({2, 13}): 0.05, frozenset({3, 13}): 0.05,
    frozenset({2, 15}): 0.10, frozenset({3, 15}): 0.10, frozenset({4, 15}): 0.05,
    frozenset({0, 15}): 0.10, frozenset({12, 2}): 0.25, frozenset({12, 13}): 0.25,
}
LEVANTINE_INTRA = {
    frozenset({5, 6}): 0.05, frozenset({5, 7}): 0.10, frozenset({6, 7}): 0.10,
    frozenset({5, 11}): 0.15,
}
MAGHREBI_INTRA = {
    frozenset({9, 14}): 0.05, frozenset({9, 16}): 0.15, frozenset({14, 16}): 0.10,
    frozenset({9, 17}): 0.20, frozenset({14, 17}): 0.20, frozenset({16, 17}): 0.05,
}
MASRI_INTRA = {frozenset({1, 10}): 0.10}

INTER_CLUSTER = {
    frozenset({0, 1}): 0.50, frozenset({0, 2}): 0.55, frozenset({0, 3}): 0.65,
    frozenset({0, 4}): 0.85, frozenset({1, 2}): 0.45, frozenset({1, 3}): 0.55,
    frozenset({1, 4}): 0.80, frozenset({2, 3}): 0.40, frozenset({2, 4}): 0.80,
    frozenset({3, 4}): 0.65,
}
INTRA_CLUSTER_DEFAULTS = {
    0: KHALEEJI_INTRA, 1: {}, 2: LEVANTINE_INTRA, 3: MASRI_INTRA, 4: MAGHREBI_INTRA,
}
INTRA_CLUSTER_FALLBACK = {0: 0.15, 1: 0.0, 2: 0.15, 3: 0.15, 4: 0.20}
INTER_CLUSTER_FALLBACK = 0.70

NADI_5_DISTANCES = {
    frozenset({0, 1}): 0.55, frozenset({0, 2}): 0.60, frozenset({0, 3}): 0.70,
    frozenset({0, 4}): 0.90, frozenset({1, 2}): 0.50, frozenset({1, 3}): 0.60,
    frozenset({1, 4}): 0.85, frozenset({2, 3}): 0.45, frozenset({2, 4}): 0.85,
    frozenset({3, 4}): 0.70,
}

AMGHADHASAN_5_DISTANCES = {
    frozenset({0, 3}): 0.10, frozenset({0, 1}): 0.35, frozenset({1, 4}): 0.40,
    frozenset({1, 3}): 0.40, frozenset({0, 4}): 0.55, frozenset({0, 2}): 0.75,
    frozenset({1, 2}): 0.70, frozenset({3, 2}): 0.75, frozenset({4, 2}): 0.70,
    frozenset({3, 4}): 0.65,
}

_AMGHADHASAN_CLUSTER = {0: 0, 1: 0, 2: 1, 3: 0, 4: 2}


def _lookup(d: Dict[frozenset, float], key: frozenset, fallback: float) -> float:
    return d.get(key, fallback)


def _build_nadi_18_distance() -> np.ndarray:
    N = 18
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            ci, cj = NADI_18_CLUSTER[i], NADI_18_CLUSTER[j]
            if ci == cj:
                intra = INTRA_CLUSTER_DEFAULTS[ci]
                d = _lookup(intra, frozenset({i, j}), INTRA_CLUSTER_FALLBACK[ci])
            else:
                d = _lookup(INTER_CLUSTER, frozenset({ci, cj}), INTER_CLUSTER_FALLBACK)
            D[i, j] = D[j, i] = d
    return D


def _build_nadi_5_distance() -> np.ndarray:
    N = 5
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = _lookup(NADI_5_DISTANCES, frozenset({i, j}), 0.70)
            D[i, j] = D[j, i] = d
    return D


def _build_amgadhasan_distance() -> np.ndarray:
    N = 5
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = _lookup(AMGHADHASAN_5_DISTANCES, frozenset({i, j}), 0.70)
            D[i, j] = D[j, i] = d
    return D


_NADI_18_DIST = _build_nadi_18_distance()
_NADI_5_DIST = _build_nadi_5_distance()
_AMGHADHASAN_DIST = _build_amgadhasan_distance()


DIALECT_DISTANCE_MATRICES: Dict[str, Tuple[np.ndarray, List[str]]] = {
    "nadi_18": (_NADI_18_DIST, [
        "OM", "SD", "SA", "KW", "QA", "LB", "JO", "SY",
        "IQ", "MA", "EG", "PL", "YE", "BH", "DZ", "AE", "TN", "LY",
    ]),
    "nadi_5": (_NADI_5_DIST, ["Khaleeji", "Iraqi", "Levantine", "Masri", "Maghrebi"]),
    "amgadhasan_5": (_AMGHADHASAN_DIST, ["EG", "LY", "LB", "SD", "MA"]),
}


def dialectal_distance(c1: int, c2: int, dataset: str) -> float:
    if dataset not in DIALECT_DISTANCE_MATRICES:
        raise ValueError(f"Unknown dataset {dataset}")
    D, _ = DIALECT_DISTANCE_MATRICES[dataset]
    if c1 < 0 or c2 < 0 or c1 >= D.shape[0] or c2 >= D.shape[1]:
        raise IndexError(f"class index out of range for {dataset}: ({c1}, {c2})")
    return float(D[c1, c2])


def similarity_matrix(dataset: str, tau: float = 0.5) -> np.ndarray:
    if dataset not in DIALECT_DISTANCE_MATRICES:
        raise ValueError(f"Unknown dataset {dataset}")
    D, _ = DIALECT_DISTANCE_MATRICES[dataset]
    S = np.exp(-D / tau)
    S = S / S.sum(axis=1, keepdims=True)
    return S


def cluster_id(class_idx: int, dataset: str) -> int:
    if dataset == "nadi_18":
        return NADI_18_CLUSTER[class_idx]
    elif dataset == "nadi_5":
        return class_idx
    elif dataset == "amgadhasan_5":
        return _AMGHADHASAN_CLUSTER.get(class_idx, -1)
    else:
        raise ValueError(dataset)


if __name__ == "__main__":
    for ds in ["nadi_18", "nadi_5", "amgadhasan_5"]:
        D, names = DIALECT_DISTANCE_MATRICES[ds]
        print(f"=== {ds} ({len(names)} classes) ===")
        print(f"  mean: {D.mean():.3f}, max: {D.max():.3f}, "
              f"min off-diag: {D[D > 0].min():.3f}")
        for i, n in enumerate(names):
            nearest = sorted([(D[i, j], names[j]) for j in range(len(names)) if j != i])[:2]
            nearest_str = ", ".join(["{} ({:.2f})".format(n2, d) for d, n2 in nearest])
            print("  {:>10}: nearest = {}".format(n, nearest_str))
