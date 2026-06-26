"""Training utilities for self-trained detectors (proof-of-concept).

Currently supports FWA: real faces + on-the-fly synthetic warping-artifact fakes.
"""
from idtest.training.fwa_aug import generate_fwa_negative, augment, to_tensor  # noqa: F401
from idtest.training.dataset import FWADataset  # noqa: F401
from idtest.training.train import train_model  # noqa: F401
