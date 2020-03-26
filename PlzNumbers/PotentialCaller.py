import numpy as np, os
from ..RynUtils import CLoader, ParameterManager

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
    def __init__(self, potential, *ignore,
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

    @property
    def lib(self):
        """

        :return:
        :rtype: module
        """
        if self._lib is None:
            loader = CLoader("PlzNumbers", os.path.dirname(os.path.abspath(__file__)),
                             source_files=["PlzNumbers.cpp", "Potators.cpp", "PyAllUp.cpp"]
                             )
            self._lib = loader.load()
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
        poots = self.lib.rynaLovesPootsLots(atoms,
                                            np.ascontiguousarray(walker).astype(float),
                                            self.potential,
                                            self.bad_walkers_file,
                                            self.error_value,
                                            self.raw_array_potential,
                                            self.vectorized_potential,
                                            self.mpi_manager,
                                            extra_bools,
                                            extra_ints,
                                            extra_floats
                                            )
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
            if isinstance(a, int):
                extra_ints.append(a)
            elif isinstance(a, bool):
                extra_bools.append(a)
            elif isinstance(a, float):
                extra_floats.append(a)

        if not isinstance(walkers, np.ndarray):
            walkers = np.array(walkers)

        if walkers.ndim == 2:
            poots = self.call_single(walkers, atoms, extra_bools, extra_ints, extra_floats)
        elif walkers.ndim == 3 or walkers.ndim == 4:
            poots = self.call_multiple(walkers, atoms, extra_bools, extra_ints, extra_floats)
        else:
            raise ValueError(
                "{}: caller expects either a single configuration, a vector of configurations, or a vector of vector of configurations".format(
                    type(self).__name__
                    )
                )

        return poots