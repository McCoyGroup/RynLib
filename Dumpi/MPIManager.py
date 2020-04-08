"""
MPIManager provides a single object-oriented interface to the stuff to initialize MPI and finalize it and all that
Binds the Scatter and Gather methods to itself so that it can be passed through the C++ code instead of every bit of code
needing to know about MPI.
Allows for easier separation of components.
"""
from ..RynUtils import CLoader
import os, sys

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
    _lib = None

    def __init__(self):
        self.init_MPI()

    @classmethod
    def _load_lib(cls):
        from ..Interface import RynLib
        import shutil
        mpi_dir = RynLib.get_conf().mpi_dir
        if not os.path.exists(os.path.join(mpi_dir, "Dumpi")):
            shutil.copytree(
                os.path.dirname(__file__),
                os.path.join(mpi_dir, "Dumpi")
                )
        loader = CLoader("Dumpi",
                         os.path.join(mpi_dir, "Dumpi"),
                         linked_libs=["mpi"],
                         include_dirs=[
                             os.path.join(mpi_dir, "lib"),
                             os.path.join(mpi_dir, "include")
                            ]
                         )
        return loader.load()

    @classmethod
    def _remove_lib(cls):
        from ..Interface import RynLib
        import shutil
        mpi_dir = RynLib.get_conf().mpi_dir
        dump_lib = os.path.join(mpi_dir, "Dumpi")
        if os.path.exists(dump_lib):
            shutil.rmtree(dump_lib)

    @property
    def lib(self):
        if self._lib is None:
            cls = type(self)
            cls._lib = cls._load_lib()
        return self._lib

    def init_MPI(self):
        cls = type(self)
        if not cls._initted:
            giveMePI = self.lib.giveMePI
            # giveMePI _must_ be called from the class
            world_rank, world_size = giveMePI(cls) # as written
            cls._world_size = world_size
            cls._world_rank = world_rank
            cls._initted = True

    def finalize_MPI(self):
        cls = type(self)
        if not cls._final:
            noMorePI = self.lib.noMorePI
            noMorePI()
            cls._final = True

    def wait(self):
        cls = type(self)
        if not cls._final:
            holdMyPI = self.lib.holdMyPI
            holdMyPI()

    @property
    def world_size(self):
        self.init_MPI()
        if self._initted is None:
            MPIManagerError.raise_uninitialized()
        return self._world_size

    @property
    def world_rank(self):
        self.init_MPI()
        if self._initted is None:
            MPIManagerError.raise_uninitialized()
        return self._world_rank

    @property
    def comm(self):
        self.init_MPI()
        if self._initted is None:
            MPIManagerError.raise_uninitialized()
        return self.lib._COMM_WORLD

    @property
    def gather(self):
        self.init_MPI()
        if self._initted is None:
            MPIManagerError.raise_uninitialized()
        return self.lib._GATHER_WALKERS

    @property
    def scatter(self):
        self.init_MPI()
        if self._initted is None:
            MPIManagerError.raise_uninitialized()
        mod = sys.modules[type(self).__module__]
        return self.lib._SCATTER_WALKERS

class MPIManagerError(Exception):
    @classmethod
    def raise_uninitialized(cls):
        raise cls("MPI must be initialized first")

class MPIManagerLoader:
    def __init__(self):
        self._manager_initialized = False
        self.manager = None
    def load_manager(self):
        if not self._manager_initialized:
            self._manager_initialized = True
            self.manager = MPIManagerObject()
            try:
                self.manager.init_MPI()
            except ImportError:
                self.manager = None
        return self.manager
    def __call__(self):
        return self.load_manager()
MPIManager = MPIManagerLoader()


