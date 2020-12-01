"""
Provides a general purpose manager for adding/removing/compiling potentials
"""

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
        self.manager.add_config(name, config_file=config_file, **opts)
        try:
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
        except:
            if name in self.list_potentials():
                self.remove_potential(name)
            raise

    def potential_config(self, name):
        self.check_potential(name)
        return self.manager.load_config(name)

    def potential_module_file(self, name):
        import glob

        self.check_potential(name)
        pot_dir = self.manager.config_loc(name)
        glob = glob.glob(os.path.join(pot_dir, "*.so")) # Unix only, but also in a container so...meh
        mod_file=None
        for f in glob:
            if name+"." in f:
                mod_file=f
                break
        return mod_file

    def load_potential(self, name):
        if name == "entos" and "entos" not in self.list_potentials():
            from ..Interface import PotentialInterface
            PotentialInterface.configure_entos()
        self.check_potential(name)
        conf = self.manager.load_config(name)
        params = conf.opt_dict
        out_dir = self.manager.config_loc(name)
        params['out_dir'] = out_dir
        return Potential.from_options(**params)

    def compile_potential(self, name, recompile=False):
        if recompile:
            mod_file = self.potential_module_file(name)
            if mod_file is not None:
                os.remove(mod_file)
        pot = self.load_potential(name)
        pot.caller # causes the potential to compile what needs to be compiled

    def potential_compiled(self, name):
        import glob
        return len(glob.glob(os.path.join(self.manager.config_loc(name), "*.so")))>0

    def import_potential(self, name, src, format='zip'):
        import tempfile as tf

        if os.path.exists(self.manager.config_loc(name)):
            raise IOError("A potential with name '{}' already exists".format(name))
        with tf.TemporaryDirectory as tmp_dir:
            shutil.unpack_archive(src, format, tmp_dir)
            extract_dir = os.path.join(tmp_dir, name)
            if not os.path.exists(extract_dir):
                raise IOError("Directory '{}' wasn't in '{}".format(name, src))
            os.rename(extract_dir, os.path.dirname(self.manager.config_loc(name)))

    def export_potential(self, name, dest, format='zip'):
        import tempfile as tf

        if os.path.exists(dest):
            raise IOError("Can't export to '{}', path exists".format(dest))

        self.check_potential(name)
        with tf.TemporaryDirectory as tmp_dir:
            src = os.path.dirname(self.manager.config_loc(name))
            arch = shutil.make_archive(src, format, tmp_dir)
            os.rename(arch, dest)

    def test_potential(self, name, input_file=None,
                       coordinates = None,
                       parameters = None,
                       atoms = None
                       ):
        import numpy as np

        pdir = self.manager.config_loc(name)
        curdir = os.getcwd()
        pot = None
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

            pot.bind_atoms(atoms)
            pot.bind_arguments(params)

            return pot(walkers)
        finally:
            if pot is not None:
                pot.clean_up()
            os.chdir(curdir)

    def test_potential_serial(self,
                              name,
                               input_file=None,
                               coordinates=None,
                               parameters=None,
                               atoms=None,
                               walkers_per_core=8,
                               time_step=1,
                               # displacement_radius=.5,
                               iterations=1,
                               steps_per_call=5,
                               print_walkers=False,
                               random_seed=None,
                               copy_geometry=False,
                               **opts
                               ):
        import numpy as np, time
        from ..DoMyCode import WalkerSet

        pdir = self.manager.config_loc(name)
        curdir = os.getcwd()
        pot = None
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

            if 'steps_per_call' in cfig:
                steps_per_call = cfig['steps_per_call']
            if 'walkers_per_core' in cfig:
                walkers_per_core = cfig['walkers_per_core']
            if 'random_seed' in cfig:
                walkers_per_core = cfig['random_seed']
            if 'copy_geometry' in cfig:
                copy_geometry = cfig['copy_geometry']
            if 'time_step' in cfig:
                time_step = cfig['time_step']
            if 'iterations' in cfig:
                iterations = cfig['iterations']

            walkers = coordinates if coordinates is not None else cfig["coordinates"]
            if isinstance(walkers, str):
                walkers = np.load(walkers)

            try:
                params = parameters if parameters is not None else cfig["parameters"]
            except (AttributeError, KeyError):
                params = []

            atoms = atoms if atoms is not None else cfig["atoms"]

            pot.bind_atoms(atoms)
            pot.bind_arguments(params)

            w = WalkerSet(
                atoms=atoms,
                initial_walker=walkers,
                mpi_manager=None,
                walkers_per_core=walkers_per_core
            )
            w.initialize(time_step)

            print("Testing Serially Over: {}".format(w))
            #
            # randomly permute things
            #
            if copy_geometry:
                testWalkersss = np.broadcast_to(w.coords, (steps_per_call,) + w.coords.shape)
            else:
                if random_seed is not None:
                    np.random.seed(random_seed)
                testWalkersss = w.get_displaced_coords(steps_per_call)
                if isinstance(testWalkersss, tuple):
                    testWalkersss = testWalkersss[0]
            test_iterations = iterations
            test_results = np.zeros((test_iterations,))
            lets_get_going = time.time()
            nsteps = steps_per_call

            #
            # run tests
            #
            pot.mpi_manager = None
            test_results_for_real = np.zeros((test_iterations, nsteps, w.num_walkers))
            for ttt in range(test_iterations):
                t0 = time.time()
                # call the potential
                # print(testAtoms)
                test_result = pot(testWalkersss)
                # then we gotta transpose back to the input layout
                test_results_for_real[ttt] = test_result
                test_results[ttt] = time.time() - t0
            gotta_go_fast = time.time() - lets_get_going

            test_result = test_results_for_real[0]
            print(
                "Mean energy {}".format(np.average(test_result)),
                sep="\n"
            )
            print("Total time: {}s (over {} iterations)".format(gotta_go_fast, test_iterations))
            print("Average total: {}s Average time per walker: {}s".format(
                np.average(test_results),
                np.average(test_results) / w.num_walkers / nsteps)
            )

            return test_results
        finally:
            if pot is not None:
                pot.clean_up()
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
        pot = None
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

            for k in (
                    'walkers_per_core', 'displacement_radius',
                    'iterations', 'steps_per_call', 'print_walkers',
                    'random_seed', 'copy_geometry'
            ):
                if k not in opts:
                    try:
                        v = cfig[k]
                    except (AttributeError, KeyError):
                        pass
                    else:
                        opts[k] = v
            pot.bind_arguments(params)
            return self._test_potential_mpi(pot, walkers, atoms, **opts)
        finally:
            if pot is not None:
                pot.clean_up()
            os.chdir(curdir)

    def _test_potential_mpi(
                cls,
                potential,
                testWalker,
                testAtoms,
                *extra_args,
                walkers_per_core=8,
                time_step=1,
                # displacement_radius=.5,
                iterations=1,
                steps_per_call=5,
                print_walkers=False,
                random_seed=None,
                copy_geometry=False
                ):
        import numpy as np, time
        from ..Dumpi import MPIManager, MPIManagerObject
        from ..DoMyCode import WalkerSet

        mpi_manager = MPIManager()

        # if mpi_manager is None:
        #     raise ImportError("MPI isn't active?")

        mpi = mpi_manager  # type: MPIManagerObject

        #
        # set up MPI
        #
        if random_seed is not None:
            np.random.seed(random_seed)
        who_am_i = mpi.world_rank if mpi is not None else 0
        w = WalkerSet(
            atoms=testAtoms,
            initial_walker=testWalker,
            mpi_manager=mpi_manager,
            walkers_per_core=walkers_per_core
        )
        w.initialize(time_step)

        if who_am_i == 0:
            print("MPIManager: {}".format(mpi))
            print("Walkers: {}".format(w))

        #
        # randomly permute things
        #
        if copy_geometry:
            testWalkersss = np.broadcast_to(w.coords, (steps_per_call,) + w.coords.shape)
        else:
            testWalkersss = w.get_displaced_coords(steps_per_call)
            if isinstance(testWalkersss, tuple):
                testWalkersss = testWalkersss[0]
        test_iterations = iterations
        test_results = np.zeros((test_iterations,))
        lets_get_going = time.time()
        nsteps = steps_per_call

        #
        # run tests
        #
        potential.mpi_manager = mpi_manager
        test_results_for_real = np.zeros((test_iterations, nsteps, w.num_walkers))
        for ttt in range(test_iterations):
            t0 = time.time()
            # call the potential
            # print(testAtoms)
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
                    "Got back result with shape {}\n and mean energy {}".format(test_result.shape, np.average(test_result)),
                    sep="\n"
                )
            print("Total time: {}s (over {} iterations)".format(gotta_go_fast, test_iterations))
            print("Average total: {}s Average time per walker: {}s".format(
                np.average(test_results),
                np.average(test_results) / w.num_walkers / nsteps)
            )
            mpi_manager.finalize_MPI()
        return test_results

    def export_potential(self, name, path):
        shutil.copytree(self.manager.config_loc(name), path)