"""
Provides a simple interface for distributing data, interacting with MPI, and handling world_size calculations in a hybrid parallel model
"""

from .MPIManager import *

__all__ = []
from .MPIManager import __all__ as exposed
__all__ += exposed