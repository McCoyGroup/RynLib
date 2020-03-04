import numpy as np

class PotentialCaller:
    """
    Takes a pointer to a C++ potential calls this potential on a set of geometries / atoms / whatever
    """
    def __init__(self, potential, *ignore,
                 bad_walker_file="bad_walkers.txt",
                 mpi_manager=None,
                 raw_array_potential = False,
                 vectorized_potential = False,
                 error_value = 10.e9
                 ):
        if len(ignore) > 0:
            raise ValueError("Only one positional argument (for the Potential) accepted")
        self.potential = potential
        self.bad_walkers_file = bad_walker_file
        self.mpi_manager = mpi_manager
        self.vectorized_potential = vectorized_potential
        self.raw_array_potential = raw_array_potential
        self.error_value = error_value

    def call_single(self, walker, atoms):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :return:
        :rtype:
        """
        from .lib import ryanLovesPoots
        return ryanLovesPoots(
            atoms,
            walker,
            self.potential.pointer,
            self.bad_walkers_file,
            self.error_value,
            self.raw_array_potential
        )

    def call_multiple(self, walker, atoms):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :return:
        :rtype:
        """
        from .lib import ryanLovesPootsLots
        smol_guy = walker.ndim == 3
        if smol_guy:
            walker = np.reshape(walker, walker.shape[:1] + (1,) + walker.shape[1:])
        poots = ryanLovesPootsLots(atoms,
                                   np.ascontiguousarray(walker),
                                   self.potential.pointer,
                                   self.bad_walkers_file,
                                   self.error_value,
                                   self.raw_array_potential,
                                   self.vectorized_potential,
                                   self.mpi_manager
                                   )
        return np.squeeze(poots)

    def __call__(self, walkers, atoms):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :return:
        :rtype:
        """
        if walkers.ndim == 2:
            poots = self.call_single(walkers, atoms)
        elif walkers.ndim == 3 or walkers.ndim == 4:
            poots = self.call_multiple(walkers, atoms)
        else:
            raise ValueError(
                "{}: caller expects either a single configuration, a vector of configurations, or a vector of vector of configurations".format(
                    type(self).__name__
                    )
                )

        return poots