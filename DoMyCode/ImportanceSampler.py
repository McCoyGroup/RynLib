"""
Provides classes for adding importance sampling to a simulation / managing importance samplers
"""

import numpy as np
from ..PlzNumbers import Potential
from ..RynUtils import Constants

__all__ = [
    "ImportanceSampler",
]

class ImportanceSampler:
    """
    A general-purpose importance sampler that applies acceptance/rejection criteria and computes local energies
    """

    def __init__(self, trial_wavefunctions, derivs=None, name=None, dx=1e-3):
        self.name = name
        self.trial_wvfn=trial_wavefunctions
        self.derivs=derivs
        self._psi=None
        self.sigmas = None
        self.time_step = None
        self.parameters = None
        self.dummied = False
        self.atomic_units = None
        self.dx = dx
        self.caller = Potential(
            name=name,
            python_potential=trial_wavefunctions
        )
    def __repr__(self):
        return "ImportanceSampler('{}', trial_wavefunction={}, derivs={})".format(
            self.name,
            self.trial_wvfn,
            self.derivs
        )

    def init_params(self, sigmas, time_step, mpi_manager, atoms, *extra_args, atomic_units=False):
        """

        :param sigmas:
        :type sigmas:
        :param time_step:
        :type time_step:
        :param mpi_manager:
        :type mpi_manager: None | MPIMangerObject
        :param atoms:
        :type atoms: Iterable[str]
        :param extra_args:
        :type extra_args:
        :return:
        :rtype:
        """
        self.sigmas = np.broadcast_to(sigmas[:, np.newaxis], sigmas.shape + (3,))
        self.time_step = time_step
        self.atomic_units = atomic_units
        if mpi_manager is not None:
            world_rank = mpi_manager.world_rank
            self.dummied = world_rank > 0
        self.caller.mpi_manager = mpi_manager
        self.caller.bind_atoms(atoms)
        # atomic_units is always the _final_ extra bool passed in
        self.caller.bind_arguments(extra_args + (atomic_units,))
    def clean_up(self):
        self.caller.clean_up()

    @property
    def mpi_manager(self):
        return self.caller.mpi_manager
    @mpi_manager.setter
    def mpi_manager(self, m):
        self.caller.mpi_manager = m
    @property
    def psi(self):
        return self._psi

    def setup_psi(self, crds):
        """
        Sets up

        :param crds:
        :type crds:
        :return:
        :rtype:
        """
        if self._psi is None:
            self._psi = np.empty(crds.shape[:2] + (3,) + crds.shape[2:], dtype=float)

    def accept(self, coords, disp):
        """
        Acceptance/Rejection of a step based on the drift term

        :param coords:
        :type coords:
        :param disp:
        :type disp:
        :return:
        :rtype:
        """

        if self.dummied:
            # gotta do the appropriate number of MPI calls, but don't want to actually compute anything
            psi1 = None
            for i in range(2):
                if self.derivs is None:
                    psi1 = self.psi_calc(coords)
                else:
                    der = self.derivs[0](coords)
            accept = None
        else:
            fx, psi1 = self.drift(coords)
            sigma = self.sigmas
            d = sigma**2/2.*fx
            if not self.atomic_units:
                d = Constants.convert(d, "angstroms", in_AU=False)
            new = coords + disp + d
            fy, psi2 = self.drift(new)
            a = self.metropolis(fx, fy, coords, new, psi1, psi2)
            check = np.random.random(size=len(coords))
            accept = np.argwhere(a > check)
            coords[accept] = new[accept]
            psi1[accept] = psi2[accept]

        num_rej = len(coords) - len(accept) if accept is not None else None
        return coords, psi1, num_rej

    def accept_step(self, step_no, coords, disp):
        coords, psi, accept = self.accept(coords, disp)
        if not self.dummied:
            self._psi[step_no] = psi
        return coords, accept

    def drift(self, coords, dx=None):
        """
        Calcuates the drift term by doing a numerical differentiation

        :param coords:
        :type coords:
        :param dx:
        :type dx:
        :return:
        :rtype:
        """

        psi = None
        der = None
        if self.dummied:
            # gotta do the appropriate number of MPI calls, but don't want to actually compute anything
            self.psi_calc(coords, dx=dx)
        
        else:
            if dx is None:
                dx = self.dx

            if self.derivs is None:
                psi = self.psi_calc(coords, dx=dx)
                der = (psi[:, 2] - psi[:, 0]) / dx / psi[:, 1]
            else:
                psi = None
                der = self.derivs[0](coords)
        return der, psi

    def psi_calc(self, coords, dx=None):
        """
        Calculates the trial wavefunction over the three displacements that are used in numerical differentiation

        :param coords:
        :type coords:
        :param trial_wvfn:
        :type trial_wvfn:
        :param dx:
        :type dx:
        :return:
        :rtype:
        """

        if dx is None:
            dx = self.dx

        trial_wvfn = self.caller
        much_psi = trial_wvfn(coords)

        if self.dummied:
            for atom_label in range(coords.shape[-2]):
                for xyz in range(3):
                    trial_wvfn(coords)
                    trial_wvfn(coords)
                    # coords[:, atom_label, xyz] -= dx
                    # much_psi[:, 0, atom_label, xyz] = trial_wvfn(coords)
                    # coords[:, atom_label, xyz] += 2. * dx
                    # much_psi[:, 2, atom_label, xyz] = trial_wvfn(coords)
                    # coords[:, atom_label, xyz] -= dx
        else:
            much_dims = much_psi.ndim
            ndims = self._psi[0].ndim
            # Assumes that much_psi is a vector of values
            # this can get fucked with the PotentialCaller (potentially)
            for i in range(ndims - much_dims):
                much_psi = np.expand_dims(much_psi, axis=-1)
            
            much_psi = np.copy(np.broadcast_to(much_psi, self._psi[0].shape))

            if not self.atomic_units:
                # _only_ these displacements need to be done in Angstroms if we're not working in A.U.
                dx = Constants.convert(dx, "angstroms", in_AU=False)
            for atom_label in range(coords.shape[-2]):
                for xyz in range(3):
                    coords[..., atom_label, xyz] -= dx
                    much_psi[..., 0, atom_label, xyz] = trial_wvfn(coords)
                    coords[..., atom_label, xyz] += 2. * dx
                    much_psi[..., 2, atom_label, xyz] = trial_wvfn(coords)
                    coords[..., atom_label, xyz] -= dx
        return much_psi

    def metropolis(self, Fqx, Fqy, x, y, psi1, psi2):
        """
        Computes the metropolis step

        :param Fqx:
        :type Fqx:
        :param Fqy:
        :type Fqy:
        :param x:
        :type x:
        :param y:
        :type y:
        :param psi1:
        :type psi1:
        :param psi2:
        :type psi2:
        :return:
        :rtype:
        """
        # takes a single timesteps worth of coordinates rather than multiple

        psi_1 = psi1[:, 1, 0, 0]
        psi_2 = psi2[:, 1, 0, 0]
        psi_ratio = (psi_2 / psi_1) ** 2
        sigma = self.sigmas
        if not self.atomic_units:
            y = Constants.convert(y, "angstroms", in_AU=True)
            x = Constants.convert(x, "angstroms", in_AU=True)
        a = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x)))
        a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
        return a

    def local_kin(self, coords, dx=None):
        """
        Calculates the local kinetic energy

        :param time_step:
        :type time_step:
        :param psi:
        :type psi:
        :param sigmas:
        :type sigmas:
        :param dx:
        :type dx:
        :return:
        :rtype:
        """
        # only thing that takes all coords at once

        if dx is None:
            dx = self.dx

        sigma = self.sigmas
        time_step = self.time_step
        if self.derivs is None:
            if not self.dummied:
                # numerical second derivative
                psi = self._psi
                d2psidx2 = ((psi[:, :, 0] - 2. * psi[:, :, 1] + psi[:, :, 2]) / dx ** 2) / psi[:, :, 1]
            else:
                d2psidx2 = np.zeros(self._psi[:, :, 1].shape)
        else:
            d2psidx2 = self.derivs[1](coords)
        # kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / time_step * d2psidx2, axis=2), axis=2)
        kin = -1. / 2 * np.tensordot(sigma**2/time_step, d2psidx2, axes=[[0, 1], [-2, -1]])
        return kin
