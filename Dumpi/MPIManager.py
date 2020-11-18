"""
MPIManager provides a single object-oriented interface to the stuff to initialize MPI and finalize it and all that
Binds the Scatter and Gather methods to itself so that it can be passed through the C++ code instead of every bit of code
needing to know about MPI.
Allows for easier separation of components.
"""

from ..RynUtils import CLoader
import multiprocessing as mp
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

    def __init__(self,
                 hybrid_parallelization = None,
                 finalize_on_exit=True
                 ):
        from ..Interface import RynLib
        parallel_threads=RynLib.flags["OpenMPThreads"]
        if not parallel_threads:
            parallel_threads = RynLib.flags["TBBThreads"]
        if parallel_threads is True:
            parallel_threads = mp.cpu_count()
        self.parallel_threads = parallel_threads
        self._hybrid_parallelization = hybrid_parallelization
        self.finalize_on_exit=finalize_on_exit
        self.init_MPI()

    @classmethod
    def _load_lib(cls):
        from ..Interface import RynLib
        mpi_dir = RynLib.mpi_dir#"/mpi"#cf.mpi_dir
        loader = CLoader("Dumpi",
                         os.path.dirname(os.path.abspath(__file__)),
                         linked_libs=["mpi"],
                         extra_compile_args=['-fopenmp', '-std=c++11'],
                         include_dirs=[
                             os.path.join(mpi_dir, "lib"),
                             os.path.join(mpi_dir, "include")
                            ]
                         )
        return loader.load()

    def __repr__(self):
        return "{}(rank={}, world_size={}, threads={}, hybrid={}, hybrid_world_size={})".format(
            type(self).__name__,
            self.world_rank,
            self.world_size,
            self.parallel_threads,
            self.hybrid_parallelization,
            self.hybrid_world_size
        )

    @classmethod
    def _remove_lib(cls):
        dump_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dumpi.so")
        if os.path.exists(dump_lib):
            os.remove(dump_lib)

    @property
    def lib(self):
        if self._lib is None:
            cls = type(self)
            cls._lib = cls._load_lib()
        return self._lib

    @property
    def hybrid_parallelization(self):
        if self._hybrid_parallelization is None:
            #world_size is smaller than the number of cores on a given node
            # or not divisible
            world_size = self.world_size
            cores_per_node = self.parallel_threads
            hybrid = world_size < cores_per_node or (world_size % cores_per_node != 0)
            self._hybrid_parallelization = hybrid
        return self._hybrid_parallelization

    @property
    def hybrid_world_size(self):
        # we assume the "world_size" is the number of nodes
        return self.world_size * self.cpu_world_size

    @property
    def cpu_world_size(self):
        # if hybrid parallelization, we return the CPUs, otherwise 1
        if self.hybrid_parallelization:
            per_core = self.parallel_threads
        else:
            per_core = 1
        return per_core

    def test(self, timeout=5):
        import threading

        test_thread = threading.Thread(target=lambda s=self:s.wait())
        test_thread.start()
        test_thread.join(timeout)
        mpi_dead = test_thread.is_alive()
        if mpi_dead:
            raise MPIManagerError("wait() was called, but the other cores never caught up--MPI probably died on a different one")
        return not mpi_dead

    def init_MPI(self):
        cls = type(self)
        if not cls._initted:
            giveMePI = self.lib.giveMePI
            # giveMePI _must_ be called from the class
            world_rank, world_size = giveMePI(cls) # as written
            cls._world_size = world_size
            cls._world_rank = world_rank
            if world_rank == -1:
                self.abort()
                raise IOError("MPI failed to initialize")
            cls._initted = True

    def finalize_MPI(self):
        cls = type(self)
        if not cls._final:
            noMorePI = self.lib.noMorePI
            noMorePI()
            cls._final = True

    def abort(self):
        cls = type(self)
        if not cls._final:
            killMyPI = self.lib.killMyPI
            killMyPI()
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
        return self.lib._GATHER_POTENTIALS

    @property
    def gather_walkers(self):
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

    def __del__(self):
        if self.finalize_on_exit:
            self.finalize_MPI()

class MPIManagerError(Exception):
    @classmethod
    def raise_uninitialized(cls):
        raise cls("MPI must be initialized first")

class MPIManagerLoader:
    def __init__(self):
        self._manager_initialized = False
        self.manager = None
    def load_manager(self):
        from ..Interface import RynLib

        if not self._manager_initialized:
            self._manager_initialized = True

            hybrid_parallelization = False
            threaded_true = (RynLib.flags["OpenMPThreads"] is True or RynLib.flags["TBBThreads"] is True)
            if threaded_true or RynLib.flags["multiprocessing"]:
                hybrid_parallelization = None
            elif RynLib.flags["OpenMPThreads"] or RynLib.flags["TBBThreads"]:
                hybrid_parallelization = True

            self.manager = MPIManagerObject(hybrid_parallelization=hybrid_parallelization)
            try:
                self.manager.init_MPI()
            except ImportError:
                self.manager = None
        return self.manager
    def __call__(self):
        return self.load_manager()
MPIManager = MPIManagerLoader()


