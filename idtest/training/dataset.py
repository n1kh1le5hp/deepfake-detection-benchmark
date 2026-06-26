"""Training dataset for FWA: real faces + on-the-fly synthetic fakes.

Each real face yields two samples: the real itself (label 0) and a synthetic
FWA fake generated from it (label 1). So ``len(dataset) == 2 * len(real_paths)``.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from idtest.io_utils import read_image
from idtest.training.fwa_aug import augment, generate_fwa_negative, to_tensor


class FWADataset(Dataset):
    def __init__(self, real_paths, size: int = 256, do_augment: bool = True, seed: int = 0):
        self.paths = list(real_paths)
        self.size = int(size)
        self.do_augment = bool(do_augment)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return 2 * len(self.paths)

    def __getitem__(self, i):
        n = len(self.paths)
        is_fake = i >= n
        img = read_image(self.paths[i - n if is_fake else i])   # BGR uint8
        if is_fake:
            img = generate_fwa_negative(img, self.rng)
        if self.do_augment:
            img = augment(img, self.rng)
        return to_tensor(img, self.size), int(is_fake)
