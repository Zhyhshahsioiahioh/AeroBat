from pathlib import Path
from . import params

MODEL_DIR = Path(__file__).resolve() / '3dmodels'

__all__ = ['MODEL_DIR', 'params']

