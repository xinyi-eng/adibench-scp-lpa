"""ADIBench: A Multi-Granularity Benchmark for Few-Shot Arabic
Dialect Identification.

This package provides:
  - 3 standardized datasets: NADI 2024 18-way, NADI 2024 5-way, amgadhasan 5-city
  - Standardized evaluation protocol: N-way K-shot with 1000 episodes
  - 8 baseline implementations
  - Comparison and analysis tools

The benchmark is designed to enable reproducible comparison of
few-shot learning methods for Arabic dialect identification across:
  - Granularity (5-way coarse vs 18-way fine)
  - Domain (country-level vs city-level)
  - Sample size (440K NADI vs 147K amgadhasan)
"""
from .data import (
    NADI_18_COUNTRIES, QADI_5_NAMES, AMGHADHASAN_5_CITIES,
    AMGHADHASAN_5_NAMES,
    QADI_18_TO_5, AMGHADHASAN_CITY_TO_IDX,
    load_nadi_18way, load_nadi_5way, load_amgadhasan_5city,
    build_fewshot_episode,
    DATASETS, get_dataset, list_datasets,
)
from .protocol import EvaluationProtocol, ResultsAggregator
from .baselines import (
    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML, Random,
    ALL_BASELINES,
)
