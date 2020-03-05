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

    def get_displaced_coords(self, n=1, trial_wvfn = None):
        # accum_disp = np.cumsum(self.get_displacements(n), axis=1)
        # return np.broadcast_to(self.coords, (n,) + self.coords.shape) + accum_disp # hoping the broadcasting makes this work...

        # this is a kinda crummy way to get this, but it allows us to get our n sets of displacements
        crds = np.zeros((n,) + self.coords.shape, dtype='float')
        if trial_wvfn is not None:
            psi = np.zeros(crds.shape[0] + (3,) + crds.shape[1:])
        bloop = self.coords
        disps = self.get_displacements(n)
        for i, d in enumerate(disps): # loop over steps
            bloop = bloop + d
            if trial_wvfn is not None:
                bloop, psi[i] = self.accept(crds[i], bloop, trial_wvfn)
            crds[i] = bloop
        if trial_wvfn is not None:
            return crds, psi
        else:
            return crds

    def accept(self, coords, disp, trial_wvfn):
        fx, psi1 = self.drift(coords, trial_wvfn)
        sigma = np.broadcast_to(self.sigmas, self.sigmas.shape + (3,))
        d = sigma**2/2.*fx
        new = disp + d
        fy, psi2 = self.drift(new, trial_wvfn)
        a = self.metropolis(fx, fy, coords, new, psi1, psi2)
        check = np.random.random(size=len(coords))
        accept = np.argwhere(a > check)
        coords[accept] = new[accept]
        psi1[accept] = psi2[accept]
        return coords, psi1

    def drift(self, coords, trial_wvfn, dx=1e-3):
        psi = self.psi_calc(coords, trial_wvfn)
        der = (psi[:, 2] - psi[:, 0]) / dx / psi[:, 1]
        return der, psi

    @staticmethod
    def psi_calc(coords, trial_wvfn, dx = 1e-3):
        much_psi = np.zeros(coords.shape[0] + (3,) + coords.shape[1:])
        for atom_label in range(len(coords[0, :, 0])):
            for xyz in range(3):
                coords[:, atom_label, xyz] -= dx
                much_psi[:, 0, atom_label, xyz] = trial_wvfn(coords)
                coords[:, atom_label, xyz] += 2. * dx
                much_psi[:, 2, atom_label, xyz] = trial_wvfn(coords)
                coords[:, atom_label, xyz] -= dx
        return much_psi

    def metropolis(self, Fqx, Fqy, x, y, psi1, psi2):
        psi_1 = psi1[:, 1, 0, 0]
        psi_2 = psi2[:, 1, 0, 0]
        psi_ratio = (psi_2 / psi_1) ** 2
        sigma = np.broadcast_to(self.sigmas, self.sigmas.shape + (3,))
        a = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x)))
        a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
        return a

    def displace(self, n=1, trial_wvfn = None):
        if trial_wvfn is not None:
            coords, psi = self.get_displaced_coords(n, trial_wvfn)
        else:
            coords = self.get_displaced_coords(n)
        self.coords = coords[-1]
        if trial_wvfn is not None:
            return coords, psi
        else:
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