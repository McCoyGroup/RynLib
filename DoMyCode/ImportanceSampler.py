
import numpy as np, os, shutil
from ..RynUtils import ConfigManager, ModuleLoader, ConfigSerializer
from ..PlzNumbers import Potential

__all__ = [
    "ImportanceSampler",
    "ImportanceSamplerManager"
]

class ImportanceSampler:
    """
    A general-purpose importance sampler that applies acceptance/rejection criteria and computes local energies
    """

    def __init__(self, trial_wavefunctions, derivs=None):
        self.trial_wvfn=trial_wavefunctions
        self.derivs=derivs
        self._psi=None
        self.sigmas = None
        self.time_step = None
        self.parameters = None
        self.caller = Potential(
            python_potential=trial_wavefunctions
        )

    def init_params(self, sigmas, time_step, atoms, mpi_manager, *extra_args):
        self.sigmas = np.broadcast_to(sigmas[:, np.newaxis], sigmas.shape + (3,))
        self.time_step = time_step
        self.caller.mpi_manager = mpi_manager
        self.caller.bind_atoms(atoms)
        self.caller.bind_arguments(extra_args)

    @property
    def psi(self):
        return self._psi

    def setup_psi(self, crds):
        if self._psi is None:
            self._psi = np.empty(crds.shape[:2] + (3,) + crds.shape[2:])

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

        fx, psi1 = self.drift(coords)
        sigma = self.sigmas
        d = sigma**2/2.*fx
        new = coords + disp + d
        fy, psi2 = self.drift(new)
        a = self.metropolis(fx, fy, coords, new, psi1, psi2)
        check = np.random.random(size=len(coords))
        accept = np.argwhere(a > check)
        coords[accept] = new[accept]
        psi1[accept] = psi2[accept]
        return coords, psi1

    def accept_step(self, step_no, coords, disp):
        coords, psi = self.accept(coords, disp)
        self._psi[step_no] = psi
        return coords

    def drift(self, coords, dx=1e-3):
        """
        Calcuates the drift term by doing a numerical differentiation

        :param coords:
        :type coords:
        :param dx:
        :type dx:
        :return:
        :rtype:
        """
        if self.derivs is None:
            psi = self.psi_calc(coords)
            der = (psi[:, 2] - psi[:, 0]) / dx / psi[:, 1]
        else:
            psi = None
            der = self.derivs[0](coords)
        return der, psi

    def psi_calc(self, coords, dx = 1e-3):
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

        trial_wvfn = self.caller
        much_psi = trial_wvfn(coords)
        much_dims = much_psi.ndim
        ndims = self._psi[0].ndim
        for i in range(ndims - much_dims):
            much_psi = np.expand_dims(much_psi, axis=-1)
        much_psi = np.copy(np.broadcast_to(much_psi, self._psi[0].shape))

        for atom_label in range(coords.shape[-2]):
            for xyz in range(3):
                coords[:, atom_label, xyz] -= dx
                much_psi[:, 0, atom_label, xyz] = trial_wvfn(coords)
                coords[:, atom_label, xyz] += 2. * dx
                much_psi[:, 2, atom_label, xyz] = trial_wvfn(coords)
                coords[:, atom_label, xyz] -= dx
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
        a = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x)))
        a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
        return a

    def local_kin(self, coords, dx=1e-3):
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

        sigma = self.sigmas
        time_step = self.time_step
        if self.derivs is None:
            psi = self._psi
            d2psidx2 = ((psi[:, :, 0] - 2. * psi[:, :, 1] + psi[:, :, 2]) / dx ** 2) / psi[:, :, 1]
        else:
            d2psidx2 = self.derivs[1](coords)
        # kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / time_step * d2psidx2, axis=2), axis=2)
        kin = -1. / 2 * np.tensordot(sigma**2/time_step, d2psidx2, axes=[[0, 1], [-2, -1]])
        return kin

class ImportanceSamplerManager:
    def __init__(self, config_dir=None):
        if config_dir is None:
            from ..Interface import RynLib
            config_dir = RynLib.sampler_directory()
        self.manager = ConfigManager(config_dir)

    def list_samplers(self):
        return self.manager.list_configs()

    def remove_sampler(self, name):
        self.manager.remove_config(name)

    def add_sampler(self, name, src, config_file = None, static_source = False, test_file=None, **opts):
        self.manager.add_config(name, config_file = config_file, **opts)
        if not static_source:
            new_src = os.path.join(self.manager.config_loc(name), name)
            if os.path.isdir(src):
                shutil.copytree(src, new_src)
            else:
                shutil.copyfile(src, new_src)
        if test_file is not None:
            shutil.copyfile(test_file, os.path.join(self.manager.config_loc(name), "test.py"))
        self.manager.edit_config(name, name=name)

    def edit_sampler(self, name, **opts):
        self.manager.edit_config(name, **opts)

    def sampler_config(self, name):
        return self.manager.load_config(name)

    def load_sampler(self, name):
        conf = self.manager.load_config(name)
        mod = conf.module
        if os.path.abspath(mod) != mod:
            mod = os.path.join(self.manager.config_loc(name), name, mod)
        if (not os.path.exists(mod)) and os.path.splitext(mod)[1] == "":
            mod = mod + ".py"

        cur_dir = os.getcwd()
        try:
            os.chdir(os.path.join(self.manager.config_loc(name), name))
            if isinstance(mod, str):
                mod = ModuleLoader().load(mod, "ImportanceSamplers")
        finally:
            os.chdir(cur_dir)

        if isinstance(mod, dict):
            trial_wfs = mod["trial_wavefunction"]
            try:
                derivs = mod["derivatives"]
            except KeyError:
                derivs = None
        else:
            trial_wfs = mod.trial_wavefunction
            try:
                derivs = mod.derivatives
            except AttributeError:
                derivs = None

        return ImportanceSampler(trial_wfs, derivs=derivs)

    def test_sampler(self, name, input_file=None):
        from .WalkerSet import WalkerSet

        pdir = self.manager.config_loc(name)
        curdir = os.getcwd()
        try:
            os.chdir(pdir)
            if input_file is None:
                input_file = os.path.join(pdir, "test.py")

            sampler = self.load_sampler(name)
            cfig = ConfigSerializer.deserialize(input_file, attribute="config")

            walkers = cfig["walkers"]
            if isinstance(walkers, str):
                walkers = WalkerSet.from_file(walkers)
            elif isinstance(walkers, dict):
                walkers = WalkerSet(**walkers)

            if 'time_step' in cfig:
                time_step = cfig["time_step"]
            else:
                time_step = 1
            if 'steps_per_propagation' in cfig:
                steps_per_propagation = cfig["steps_per_propagation"]
            else:
                steps_per_propagation = 5

            walkers.initialize(time_step)

            if 'sigmas' in cfig:
                sigmas = cfig["sigmas"]
            elif isinstance(walkers, WalkerSet):
                sigmas = walkers.sigmas
            else:
                sigmas = 1

            if 'parameters' in cfig:
                parameters = cfig["parameters"]
            else:
                parameters = ()

            mpi_manager = None

            sampler.init_params(sigmas, time_step, walkers.atoms, mpi_manager, *parameters)

            # print(walkers.coords.shape)

            disp_walks = walkers.displace(steps_per_propagation, importance_sampler=sampler)
            return sampler.local_kin(disp_walks)
        finally:
            os.chdir(curdir)