"""
Single seeding entry point for the whole pipeline.

Call set_seed() at the start of every runnable script (training, sweeps,
reranking, evaluation) so that results are reproducible across runs.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every random number generator used in the pipeline.

    Seeds Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices),
    and fixes `PYTHONHASHSEED` so hash-based ordering is stable. FAISS
    draws its randomness from NumPy, so it is covered as well.

    Parameters
    ----------
    seed : int
        The seed applied to all generators (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
