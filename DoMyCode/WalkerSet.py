"""
Defines a WalkerSet to be used by a simulation
"""

import numpy as np
from ..RynUtils.Constants import Constants

__all__ = [ "WalkerSet" ]

class WalkerSet:
    def __init__(self,
                 atoms=None,
                 masses=None,
                 initial_walker=None,
                 initial_weights=1.,
                 mass_scaling=None,
                 num_walkers=None,
                 mpi_manager=None,
                 walkers_per_core=None
                 ):

        self.n = len(atoms)

        self.mpi_manager = mpi_manager
        if num_walkers is None:
            if mpi_manager is None:
                from ..Interface import RynLib
                if RynLib.flags["OpenMPThreads"] is True or RynLib.flags["multiprocessing"]:
                    import multiprocessing as mp
                    world_size = mp.cpu_count()
                elif RynLib.flags["OpenMPThreads"]:
                    world_size = RynLib.flags["OpenMPThreads"]
                else:
                    world_size = 1
                # raise TypeError("MPIManager is None (meaning MPI isn't configured) but 'num_walkers' not passed")
            elif mpi_manager.world_rank == 0:
                world_size = mpi_manager.hybrid_world_size
            else:
                world_size = mpi_manager.cpu_world_size
            num_walkers = walkers_per_core*world_size

        self.num_walkers = num_walkers

        if masses is None:
            masses = [Constants.mass(a) for a in atoms]
        masses = np.array(masses)
        if mass_scaling is not None:
            masses = masses * mass_scaling

        self.atoms = atoms
        self.masses = masses

        walker_choice_inds = None # to make sure that the walkers and weights line up
        if isinstance(initial_walker, str):
            initial_walker = np.load(initial_walker)
        initial_walker = np.asarray(initial_walker)
        if len(initial_walker.shape) == 2:
            initial_walker = np.array([initial_walker] * num_walkers)
        elif len(initial_walker) > num_walkers or len(initial_walker) < num_walkers:
            walker_choice_inds = np.random.randint(0, initial_walker.shape[0], num_walkers)
            initial_walker = initial_walker[walker_choice_inds]
        self.coords = initial_walker
        self._cached_coords = None # efficient for big-walker simulations

        if isinstance(initial_weights, str):
            initial_weights = np.load(initial_weights)
        elif isinstance(initial_weights, (float, int, np.integer, np.floating)):
            initial_weights = np.full((num_walkers,), initial_weights)
        if len(initial_weights) > num_walkers or len(initial_weights) < num_walkers:
            if walker_choice_inds is None:
                # should we raise a misconfiguration error?
                raise ValueError("Can't figure out how to map initial weights onto initial walkers?")
            initial_weights = initial_weights[walker_choice_inds]
        self.weights = initial_weights

        self.parents = np.arange(num_walkers)
        self.sigmas = None
        self._parents = self.coords.copy()
        self._parent_weights = self.weights.copy()

    def __repr__(self):
        return "{}(num_walkers={}, atoms={}, time_step={})".format(
            type(self).__name__,
            self.num_walkers,
            self.atoms,
            self.deltaT
        )

    @classmethod
    def from_file(cls, file, **opts):
        npz = np.load(file)
        return cls(atoms=npz["atoms"], masses=npz["masses"], initial_walker=npz["walkers"], **opts)

    def initialize(self, deltaT, D=1./2.):
        """Sets up necessary parameters for use in calculating displacements and stuff

        :param deltaT:
        :type deltaT:
        :param D:
        :type D:
        :return:
        :rtype:
        """
        self.deltaT = deltaT
        if self.sigmas is None:
            self.sigmas = np.sqrt((2.0 * D * deltaT) / self.masses)

    def get_displacements(self, steps = 1, coords = None, atomic_units = False):

        if coords is None:
            coords = self.coords
        shape = (steps, ) + coords.shape[:-2] + coords.shape[-1:]
        disps = np.array([
            np.random.normal(0.0, sig, size=shape) for sig in self.sigmas
        ])

        disps = np.transpose(disps, (1, 2, 0, 3))

        if not atomic_units:
            disps = Constants.convert(disps, "angstroms", in_AU=False)

        return disps

    def get_displaced_coords(self, n=1, coords=None, importance_sampler=None, atomic_units=False):
        # this is a kinda crummy way to get this, but it allows us to get our n sets of displacements
        if coords is None:
            coords = self.coords
        crds = np.empty((n,) + coords.shape, dtype=float)
        if importance_sampler is not None:
            importance_sampler.setup_psi(crds)
        bloop = coords.astype(float)
        # print("wat...?", id(self.coords))
        disps = self.get_displacements(n, coords, atomic_units=atomic_units)
        if importance_sampler is not None:
            rej = [None] * n
        else:
            rej = None
        for i, d in enumerate(disps): # loop over steps
            if importance_sampler is not None:
                if importance_sampler.atomic_units is not atomic_units:
                    raise ValueError("Importance sampler and walker set disagree on units")
                bloop, accept = importance_sampler.accept_step(i, bloop, d)
                bloop = np.copy(bloop)
                rej[i] = accept
                if importance_sampler.mpi_manager is not None:
                    importance_sampler.mpi_manager.wait()
            else:
                bloop = bloop + d
            crds[i] = bloop
        return crds, rej

    def displace(self, n=1, importance_sampler = None, atomic_units=False):
        coords, rej = self.get_displaced_coords(n, importance_sampler=importance_sampler, atomic_units=atomic_units)
        self.coords = coords[-1]
        return coords, rej

    def _setup_dw(self):
        self.parents = np.arange(self.num_walkers)
        self._parents = self.coords.copy()
        self._parent_weights = self.weights.copy()

    def descendent_weight(self):
        """Handles the descendent weighting in the system

        :return: tuple of parent coordinates, descendend weights, and original weights
        :rtype:
        """

        weights = np.array( [ np.sum(self.weights[ self.parents == i ]) for i in range(self.num_walkers) ] )
        descendent_weights = {"coords":self._parents, "weights":weights, "original_weights":self._parent_weights}

        return descendent_weights

    def snapshot(self, file):
        """Snapshots the current walker set to file"""

        np.savez(file,
                 coords=self.coords, weights=self.weights, sigmas=self.sigmas,
                 parents=self.parents,
                 parent_coords=self._parents,
                 parent_weights = self._parent_weights
                 )

    @classmethod
    def load(cls, file, atoms = None, masses = None):
        """Reloads WalkerSet from file"""

        npz = np.load(file)

        sigs = npz['sigmas']

        if masses is None:
            masses = np.ones(len(sigs))
        if atoms is None:
            atoms = ['H'] * len(sigs)

        coords = npz['coords']
        self = cls(atoms = atoms, masses = masses, initial_walker=coords, num_walkers=len(coords))

        self.parents = npz['parents']
        self.sigmas = npz['sigmas']
        self.weights = npz['weights']
        self.parents = npz['parents']
        self._parent_weights = npz['parent_weights']
        self._parent_coords = npz['parent_coords']

        return self