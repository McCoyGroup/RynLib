"""
Defines the WalkerSet to be used by the simulation
"""

import numpy as np

__all__ = [ "WalkerSet" ]

class WalkerSet:
    def __init__(self, atoms = None, masses = None, initial_walker = None, num_walkers = None):
        self.n = len(atoms)
        self.num_walkers = num_walkers

        self.atoms = atoms
        self.masses = masses

        if len(initial_walker.shape) == 2:
            initial_walker = np.array([ initial_walker ] * num_walkers)
        else:
            self.num_walkers = len(initial_walker)

        self.coords = initial_walker
        self.weights = np.ones(num_walkers)

        self.parents = np.arange(num_walkers)
        self.sigmas = None
        self._parents = self.coords.copy()
        self._parent_weights = self.weights.copy()

    def initialize(self, deltaT, D):
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
            self.sigmas = np.sqrt((2 * D * deltaT) / self.masses)

    def get_displacements(self, steps = 1, in_AU = True):
        shape = (steps, ) + self.coords.shape[:-2] + self.coords.shape[-1:]
        disps = np.array([
            np.random.normal(0.0, sig, size = shape) for sig in self.sigmas
        ])

        disps = np.transpose(disps, (1, 2, 0, 3))

        if not in_AU:
            disps = Constants.convert(disps, "angstroms", in_AU = False)

        return disps
    def get_displaced_coords(self, n=1):
        # accum_disp = np.cumsum(self.get_displacements(n), axis=1)
        # return np.broadcast_to(self.coords, (n,) + self.coords.shape) + accum_disp # hoping the broadcasting makes this work...

        # this is a kinda crummy way to get this, but it allows us to get our n sets of displacements
        crds = np.zeros((n,) + self.coords.shape, dtype='float')
        bloop = self.coords
        disps = self.get_displacements(n)
        for i, d in enumerate(disps): # loop over atoms
            bloop = bloop + d
            crds[i] = bloop

        return crds

    def displace(self, n=1):
        coords = self.get_displaced_coords(n)
        self.coords = coords[-1]
        return coords

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