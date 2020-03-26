"""
MPIManager provides a single object-oriented interface to the stuff to initialize MPI and finalize it and all that
Binds the Scatter and Gather methods to itself so that it can be passed through the C++ code instead of every bit of code
needing to know about MPI.
Allows for easier separation of components.
"""
from ..RynUtils import CLoader
from ..Interface import GeneralConfig
import os

__all__ = [
    "MPIManager",
    "MPIManagerObject"
]

class MPIManagerObject:
    _initted = False
    _final = False
    _comm = None
    _world_size = None
    _world_rank = None

    def __init__(self):
        self._lib = None
        self.init_MPI()

    @property
    def lib(self):
        if self._lib is None:
            loader = CLoader("Dumpi", os.path.dirname(os.path.abspath(__file__)),
                             linked_libs=["mpi"],
                             include_dirs=[GeneralConfig.get_conf().mpi_dir]
                             )
            self._lib = loader.load()
        return self._lib

    def init_MPI(self):
        if not self._initted:
            giveMePI = self.lib.giveMePI
            cls = type(self)
            world_size, world_rank = giveMePI(cls) # as written
            cls._world_size = world_size
            cls._world_rank = world_rank
            cls._initted = True

    def finalize_MPI(self):
        if not self._final:
            noMorePI = self.lib.noMorePI
            noMorePI()
            self._final = True

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

class MPIManagerLoader:
    _manager_initialized = False
    manager = None

    def load_manager(self):
        if not self._manager_initialized:
            self.manager = MPIManagerObject()
            try:
                self.manager.init_MPI()
            except ImportError:
                self.manager = None
            self._manager_initialized = True

        return self.manager

    def __call__(self):
        self.load_manager()
MPIManager = MPIManagerLoader()


