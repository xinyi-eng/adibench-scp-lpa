"""ALPS: Arabic Linguistic-distance Prior Sampling.

Few-shot dialect identification with linguistic-distance priors.

Three innovations:
  1. L2C-ProtoNet: Linguistic-distance-weighted prototype network
  2. Geo-Curriculum: Geographic-difficulty-based training schedule
  3. Multi-granularity evaluation: 5/18/25-way Arabic dialect tasks
"""
from .data import (
    NADI_18_COUNTRIES, AMGHADHASAN_5_CITIES,
    QADI_18_TO_5, AMGHADHASAN_TO_5,
    load_nadi_18way, load_amgadhasan_5way,
    build_fewshot_episode,
)
from .distance import (
    NADI_18_DISTANCE, QADI_5_DISTANCE, AMGHADHASAN_5_DISTANCE,
    get_distance_matrix, get_linguistic_distance,
)
from .protonet import ProtoNet, L2CProtoNet
from .curriculum import GeoCurriculum
from .train import train_fewshot
from .eval import evaluate_fewshot, multi_granularity_eval
