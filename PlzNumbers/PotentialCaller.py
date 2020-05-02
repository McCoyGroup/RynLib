import numpy as np, os, multiprocessing as mp
from ..RynUtils import CLoader

__all__ = [
    "PotentialCaller"
]

class PotentialCaller:
    """
    Takes a pointer to a C++ potential calls this potential on a set of geometries / atoms / whatever
    """
    __props__ = [
        "bad_walker_file",
        "mpi_manager",
        "raw_array_potential",
        "vectorized_potential",
        "error_value"
    ]
    def __init__(self,
                 potential, *ignore,
                 bad_walker_file="bad_walkers.txt",
                 mpi_manager=None,
                 raw_array_potential = False,
                 vectorized_potential = False,
                 error_value = 10.e9
                 ):
        if len(ignore) > 0:
            raise ValueError("Only one positional argument (for the potential) accepted")
        self.potential = potential
        self.bad_walkers_file = bad_walker_file
        self.mpi_manager = mpi_manager
        self.vectorized_potential = vectorized_potential
        self.raw_array_potential = raw_array_potential
        self.error_value = error_value
        self._lib = None
        self._py_pot = not repr(self.potential).startswith("<capsule object ")  # wow this is a hack...
        self._wrapped_pot = None

    @classmethod
    def load_lib(cls):
        loader = CLoader("PlzNumbers",
                         os.path.dirname(os.path.abspath(__file__)),
                         extra_compile_args=["-fopenmp"],
                         extra_link_args=["-fopenmp"],
                         source_files=["PlzNumbers.cpp", "Potators.cpp", "PyAllUp.cpp"]
                )
        return loader.load()

    @classmethod
    def reload(cls):
        try:
            os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PlzNumbers.so"))
        except OSError:
            pass
        return cls.load_lib()
    @property
    def lib(self):
        """

        :return:
        :rtype: module
        """
        if self._lib is None:
            self._lib = self.load_lib()
        return self._lib

    def call_single(self, walker, atoms, extra_bools=(), extra_ints=(), extra_floats=()):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :return:
        :rtype:
        """

        if self._py_pot:
            return self.potential(walker, atoms, (extra_bools, extra_ints, extra_floats))
        else:
            return self.lib.rynaLovesPoots(
                atoms,
                np.ascontiguousarray(walker).astype(float),
                self.potential,
                self.bad_walkers_file,
                self.error_value,
                self.raw_array_potential,
                extra_bools,
                extra_ints,
                extra_floats
            )

    def _mp_wrap(self,
                 pot,
                 num_walkers,
                 mpi_manager
                 ):
        # We'll provide a wrapper that we can use with our functions to add parallelization
        # based on Pool.map
        # The wrapper will first check to make sure that we _are_ using a hybrid parallelization model

        if mpi_manager is None:
            from ..Interface import RynLib
            hybrid_p = RynLib.use_MP
            world_size = mp.cpu_count()
        else:
            hybrid_p = mpi_manager.hybrid_parallelization
            if hybrid_p:
                world_size = mpi_manager.hybrid_world_size
            else:
                world_size = 0 # should throw an error if we try to compute the block_size

        if hybrid_p:
            pool = mp.Pool()
            block_size = np.math.ceil(num_walkers/world_size)
            def potential(walkers, atoms, extra_bools=(), extra_ints=(), extra_floats=()):
                blocks = np.array_split(walkers, block_size)
                p = lambda w, a=atoms, e=(extra_bools, extra_ints, extra_floats): p(a, e)
                res = pool.map(p, blocks)
                return np.concatenate(res)
        else:
            potential = pot

        return potential

    def call_multiple(self, walker, atoms, extra_bools=(), extra_ints=(), extra_floats=()):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :return:
        :rtype:
        """

        smol_guy = walker.ndim == 3
        if smol_guy:
            walker = np.reshape(walker, walker.shape[:1] + (1,) + walker.shape[1:])

        if self._py_pot and self._wrapped_pot is None:
            num_walkers = walker.shape[1]
            self._wrapped_pot = self._mp_wrap(self.potential, num_walkers, self.mpi_manager)

        if self._py_pot and self.mpi_manager is None:
            poots =  self._wrapped_pot(walker, atoms, (extra_bools, extra_ints, extra_floats))
        elif self._py_pot:
            walker = walker.transpose((1, 0, 2, 3))
            poots = self.lib.rynaLovesPyPootsLots(
                atoms,
                np.ascontiguousarray(walker).astype(float),
                self._wrapped_pot,
                self.mpi_manager,
                (extra_bools, extra_ints, extra_floats)
            )
            if poots is not None:
                poots = poots.transpose()
        else:
            walker = walker.transpose((1, 0, 2, 3))
            if self.mpi_manager is not None:
                hp = self.mpi_manager.hybrid_parallelization
            else:
                from ..Interface import RynLib
                hp = RynLib.use_MP
            print("asdasd asd",
                  extra_bools,
                  extra_ints,
                  extra_floats,
                  self.mpi_manager is None,
                  self.potential
                  )
            coords = np.ascontiguousarray(walker).astype(float)
            pot = self.potential
            poots = self.lib.rynaLovesPootsLots(
                self.mpi_manager,
                atoms,
                coords,
                pot,
                self.bad_walkers_file,
                extra_bools,
                extra_ints,
                extra_floats,
                float(self.error_value),
                bool(self.raw_array_potential),
                bool(self.vectorized_potential),
                bool(hp)
            )
            if poots is not None:
                poots = poots.transpose()
        return np.squeeze(poots)

    def __call__(self, walkers, atoms, *extra_args):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :return:
        :rtype:
        """

        extra_bools = []
        extra_ints = []
        extra_floats = []
        for a in extra_args:
            if a is True or a is False:
                extra_bools.append(a)
            elif isinstance(a, int):
                extra_ints.append(a)
            elif isinstance(a, float):
                extra_floats.append(a)

        if not isinstance(walkers, np.ndarray):
            walkers = np.array(walkers)

        ndim = walkers.ndim

        if ndim == 2:
            poots = self.call_single(walkers, atoms, extra_bools, extra_ints, extra_floats)
        elif ndim == 3 or ndim == 4:
            poots = self.call_multiple(walkers, atoms, extra_bools, extra_ints, extra_floats)
        else:
            raise ValueError(
                "{}: caller expects data of rank 2, 3, or 4. Got {}.".format(
                    type(self).__name__,
                    ndim
                    )
                )

        return poots