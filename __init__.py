# Marks fd_gan as a Python package

from .generators import BlendGenerator, TransferGenerator
from .discriminator import Discriminator
from .losses import classification_loss, forgery_similarity_loss, diversity_loss
from .training import FDGANTrainer

__version__ = "1.0.0"
__all__ = [
    "BlendGenerator",
    "TransferGenerator",
    "Discriminator",
    "classification_loss",
    "forgery_similarity_loss",
    "diversity_loss",
    "FDGANTrainer"
]
