from ..RynUtils import ConfigManager, ConfigSerializer
from .Potential import Potential
import os, shutil

__all__ = [
    "PotentialManager"
]

class PotentialManager:
    def __init__(self, config_dir=None):
        if config_dir is None:
            from ..Interface import RynLib
            config_dir = RynLib.potential_directory()
        self.manager = ConfigManager(config_dir)

    def list_potentials(self):
        return self.manager.list_configs()

    def check_potential(self, name):
        if not self.manager.check_config(name):
            raise IOError("No potential {}".format(name))
    def remove_potential(self, name):
        self.check_potential(name)
        self.manager.remove_config(name)

    def add_potential(self, name, src,
                      config_file = None, data=None,
                      test = None,
                      **opts
                      ):
        self.manager.add_config(name, config_file = config_file, **opts)
        cf =self.manager.load_config(name)
        try:
            static_source = cf.static_source
        except (KeyError, AttributeError):
            static_source = False
        try:
            python_potential = cf.python_potential
        except (KeyError, AttributeError):
            python_potential = False

        if test is not None:
            shutil.copyfile(test, os.path.join(self.manager.config_loc(name), "test.py"))

        if data is not None:
            data_src = os.path.join(self.manager.config_loc(name), os.path.basename(data))
            shutil.copytree(data, data_src)

        if not static_source:
            if python_potential:
                base_dir = self.manager.config_loc(name)
            else:
                base_dir = os.path.join(self.manager.config_loc(name), "raw_source")
            new_src = os.path.join(base_dir, os.path.basename(src))
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)
            if os.path.isdir(src):
                shutil.copytree(src, new_src)
            else:
                shutil.copyfile(src, new_src)
            self.manager.edit_config(name, name=name, potential_source=new_src, static_source=False)
        else:
            self.manager.edit_config(name, name=name, potential_source=src, static_source=True)

    def potential_config(self, name):
        self.check_potential(name)
        return self.manager.load_config(name)

    def load_potential(self, name):
        if name == "entos" and "entos" not in self.list_potentials():
            from ..Interface import PotentialInterface
            PotentialInterface.configure_entos()
        self.check_potential(name)
        conf = self.manager.load_config(name)
        params = conf.opt_dict
        out_dir = self.manager.config_loc(name)
        params['out_dir'] = out_dir
        return Potential(**params)

    def compile_potential(self, name):
        pot = self.load_potential(name)
        pot.caller # causes the potential to compile what needs to be compiled

    def potential_compiled(self, name):
        import glob
        return len(glob.glob(os.path.join(self.manager.config_loc(name), "*.so")))>0

    def test_potential(self, name, input_file=None,
                       coordinates = None,
                       parameters = None,
                       atoms = None
                       ):
        import numpy as np

        pdir = self.manager.config_loc(name)
        curdir = os.getcwd()
        try:
            os.chdir(pdir)

            if input_file is None:
                inp = os.path.join(pdir, "test.py")
                if os.path.exists(inp):
                    input_file = inp

            pot = self.load_potential(name)

            if input_file is not None:
                cfig = ConfigSerializer.deserialize(input_file, attribute="config")
            else:
                cfig = {}

            walkers = coordinates if coordinates is not None else cfig["coordinates"]
            if isinstance(walkers, str):
                walkers = np.load(walkers)

            try:
                params = parameters if parameters is not None else cfig["parameters"]
            except (AttributeError, KeyError):
                params = []

            atoms = atoms if atoms is not None else cfig["atoms"]

            return pot(walkers, atoms, *params)
        finally:
            os.chdir(curdir)

    def test_potential_mpi(self, name,
                           input_file=None,
                           coordinates=None,
                           parameters=None,
                           atoms=None,
                           **opts
                           ):
        import numpy as np

        pdir = self.manager.config_loc(name)
        curdir = os.getcwd()
        try:
            os.chdir(pdir)
            if input_file is None:
                inp = os.path.join(pdir, "test.py")
                if os.path.exists(inp):
                    input_file = inp

            pot = self.load_potential(name)

            if input_file is not None:
                cfig = ConfigSerializer.deserialize(input_file, attribute="config")
            else:
                cfig = {}

            walkers = coordinates if coordinates is not None else cfig["coordinates"]
            if isinstance(walkers, str):
                walkers = np.load(walkers)

            try:
                params = parameters if parameters is not None else cfig["parameters"]
            except (AttributeError, KeyError):
                params = []

            atoms = atoms if atoms is not None else cfig["atoms"]

            for k in ('walkers_per_core', 'displacement_radius', 'iterations', 'steps_per_call', 'print_walkers'):
                if k not in opts:
                    try:
                        v = cfig[k]
                    except (AttributeError, KeyError):
                        pass
                    else:
                        opts[k] = v

            return self._test_potential_mpi(pot, walkers, atoms, *params, **opts)
        finally:
            os.chdir(curdir)

    def _test_potential_mpi(cls,
                           potential,
                           testWalker,
                           testAtoms,
                           *extra_args,
                           walkers_per_core=8,
                           displacement_radius=.5,
                           iterations=5,
                           steps_per_call=5,
                           print_walkers=False
                           ):
        import numpy as np, time
        from ..Dumpi import MPIManager, MPIManagerObject

        mpi_manager = MPIManager()

        if mpi_manager is None:
            raise ImportError("MPI isn't installed. Use `container config install_mpi` first.")

        mpi = mpi_manager  # type: MPIManagerObject

        #
        # set up MPI
        #
        who_am_i = mpi.world_rank
        num_cores = mpi.world_size
        num_walkers_per_core = walkers_per_core
        if who_am_i == 0:
            num_walkers = num_cores * num_walkers_per_core
        else:
            num_walkers = num_walkers_per_core

        if who_am_i == 0:
            print("Number of processors / walkers: {} / {}".format(num_cores, num_walkers))

        #
        # randomly permute things
        #
        testWalkersss = np.array([testWalker] * num_walkers).astype(float)
        testWalkersss += np.random.uniform(
            low=-displacement_radius,
            high=displacement_radius,
            size=testWalkersss.shape
        )
        test_iterations = iterations
        test_results = np.zeros((test_iterations,))
        lets_get_going = time.time()
        nsteps = steps_per_call
        # we compute the same walker for each of the nsteps, but that's okay -- gives a nice clean test that everything went right
        testWalkersss = np.ascontiguousarray(np.broadcast_to(testWalkersss, (nsteps,) + testWalkersss.shape))

        #
        # run tests
        #
        potential.mpi_manager = mpi_manager
        test_results_for_real = np.zeros((test_iterations, nsteps, num_walkers))
        for ttt in range(test_iterations):
            t0 = time.time()
            # call the potential
            print(testAtoms)
            test_result = potential(
                testWalkersss,
                testAtoms,
                *extra_args
            )
            # then we gotta transpose back to the input layout
            if who_am_i == 0:
                test_results_for_real[ttt] = test_result
                test_results[ttt] = time.time() - t0
        gotta_go_fast = time.time() - lets_get_going

        #
        # tell me how you really feel
        #
        if who_am_i == 0:
            test_result = test_results_for_real[0]
            if print_walkers:
                print(
                    # "Fed in: {}".format(testWalkersss),
                    "Fed in walker array with shape {}".format(testWalkersss.shape),
                    sep="\n"
                )
                print(
                    "Got back: {}".format(test_result),
                    "  with shape {}".format(test_result.shape),
                    sep="\n"
                )
            else:
                print(
                    "Got back result with shape {}\n  and first element {}".format(test_result.shape, test_result[0]),
                    sep="\n"
                )
            print("Total time: {}s (over {} iterations)".format(gotta_go_fast, test_iterations))
            print("Average total: {}s Average time per walker: {}s".format(
                np.average(test_results),
                np.average(test_results) / num_walkers / nsteps)
            )
            mpi_manager.finalize_MPI()
        return test_results

    def export_potential(self, name, path):
        shutil.copytree(self.manager.config_loc(name), path)