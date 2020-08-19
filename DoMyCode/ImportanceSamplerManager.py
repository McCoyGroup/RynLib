"""
Provides a ConfigManager for the importance samplers
"""

import os, shutil, numpy as np
from ..RynUtils import ConfigManager, ModuleLoader, ConfigSerializer
from .ImportanceSampler import ImportanceSampler

__all__=[
    "ImportanceSamplerManager"
]

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
                mod = ModuleLoader().load(mod, "")
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

        return ImportanceSampler(trial_wfs, derivs=derivs, name=name)

    def test_sampler(self, name, input_file=None, mpi_manager=None):
        from .WalkerSet import WalkerSet
        import time

        pdir = self.manager.config_loc(name)
        curdir = os.getcwd()
        sampler = None
        try:
            os.chdir(pdir)
            if input_file is None:
                input_file = os.path.join(pdir, "test.py")

            sampler = self.load_sampler(name)
            cfig = ConfigSerializer.deserialize(input_file, attribute="config")

            walkers = cfig["walkers"]
            if isinstance(walkers, str):
                walkers = WalkerSet.from_file(walkers, mpi_manager=mpi_manager)
            elif isinstance(walkers, dict):
                walkers = WalkerSet(mpi_manager=mpi_manager, **walkers)

            if 'random_seed' in cfig:
                np.random.seed(cfig['random_seed'])

            if 'time_step' in cfig:
                time_step = cfig["time_step"]
            else:
                time_step = 1
            if 'steps_per_propagation' in cfig:
                steps_per_propagation = cfig["steps_per_propagation"]
            else:
                steps_per_propagation = 5

            walkers.initialize(time_step)

            # print(walkers)

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

            if 'atomic_units' in cfig:
                atomic_units = cfig["atomic_units"]
            else:
                atomic_units = False

            sampler.init_params(sigmas, time_step, mpi_manager, walkers.atoms, *parameters, atomic_units=atomic_units)

            # print(walkers.coords.shape)

            start = time.time()
            disp_walks = walkers.displace(steps_per_propagation, importance_sampler=sampler, atomic_units=atomic_units)
            disp_walks2 = disp_walks[-1, -1]
            ke = sampler.local_kin(disp_walks)
            end = time.time()

            tot = end-start
            av = tot / np.prod(disp_walks.shape)

            meta = {"mpi" : mpi_manager, "walkers": walkers, "timing": tot, "average": av}
            return ke, meta
        finally:
            if sampler is not None:
                sampler.clean_up()
            os.chdir(curdir)

    def test_sampler_mpi(self, name, input_file=None):
        from ..Dumpi import MPIManager
        mpi_manager = MPIManager()

        yield mpi_manager
        yield self.test_sampler(name, input_file=input_file, mpi_manager=mpi_manager)