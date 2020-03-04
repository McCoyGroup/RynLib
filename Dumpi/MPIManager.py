"""
MPIManager provides a single object-oriented interface to the stuff to initialized MPI and finialize it and all that
"""

class MPIManager:
    _initted = False
    _final = False
    _comm = None
    _world_size = None
    _world_rank = None
    def __init__(self):
        self.init_MPI()

    def init_MPI(self):
        if not self._initted:
            from .lib import *
            cls = type(self)
            world_size, world_rank = giveMePI(cls) # as written
            cls._world_size = world_size
            cls._world_rank = world_rank
            cls._initted = True

    def finalize_MPI(self):
        if not self._final:
            from .lib import *


    @property
    def world_size(self):
        if not self._initted is None:
            MPIManagerError.raise_uninitialized()
        return self._world_size

    @property
    def world_rank(self):
        if not self._initted is None:
            MPIManagerError.raise_uninitialized()
        return self._world_rank

    @property
    def comm(self):
        if not self._initted is None:
            MPIManagerError.raise_uninitialized()
        return self._comm

class MPIManagerError(Exception):
    @classmethod
    def raise_uninitialized(cls):
        raise cls("MPI must be initialized first")





