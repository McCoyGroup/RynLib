"""
The overall interface to the RynLib CLI
"""

import os, shutil
from .RynUtils import Config, ConfigSerializer
from .PlzNumbers import PotentialManager
from .DoMyCode import SimulationManager, ImportanceSamplerManager

__all__ = [
    "SimulationInterface",
    "PotentialInterface",
    "RynLib"
]

class SimulationInterface:
    """
    Defines all of the CLI options for working with simulations
    """
    @classmethod
    def list_simulations(cls):
        print("\n".join(SimulationManager().list_simulations()))

    @classmethod
    def add_simulation(self, name=None, config_file=None):
        SimulationManager().add_simulation(name, config_file)
        print("Added simulation {}".format(name))

    @classmethod
    def remove_simulation(self, name=None):
        SimulationManager().remove_simulation(name)
        print("Removed simulation {}".format(name))

    @classmethod
    def simulation_status(self, name=None):
        status = SimulationManager().simulation_ran(name)
        config = SimulationManager().simulation_config(name)
        print(
            "Status: {}".format(status),
            *("  {}: {}".format(k, v) for k, v in config.opt_dict.items()),
            sep="\n"
        )

    @classmethod
    def edit_simulation(self, name=None, **opts):
        SimulationManager().edit_simulation(name, **opts)

    @classmethod
    def export_simulation(cls, name=None, path=None):
        SimulationManager().export_simulation(name, path)

    @classmethod
    def run_simulation(self, name=None):
        print("Running simulation {}".format(name))
        SimulationManager().run_simulation(name)
        print("Finished running simulation {}".format(name))

    @classmethod
    def test_HO(cls):
        sm = SimulationManager()
        if "test_HO" in sm.list_simulations():
            sm.remove_simulation("test_HO")
        pm = PotentialManager()
        if 'HarmonicOscillator' not in pm.list_potentials():
            PotentialInterface.configure_HO()
        sm.add_simulation("test_HO",
                          os.path.join(os.path.dirname(__file__), "Tests", "TestData", "test_HO.py")
                          )
        cls.run_simulation("test_HO")

    @classmethod
    def list_samplers(cls):
        print("\n".join(ImportanceSamplerManager().list_samplers()))

    @classmethod
    def add_sampler(self, name=None, config_file=None, source=None, test_file=None):
        no_config = config_file is None
        if config_file is None:
            if os.path.exists(os.path.join(source, "config.py")):
                config_file = os.path.join(source, "config.py")
        if test_file is None:
            if os.path.exists(os.path.join(source, "test.py")):
                test_file = os.path.join(source, "test.py")
        if no_config:
            if os.path.exists(os.path.join(source, name)):
                source = os.path.join(source, name)
        ImportanceSamplerManager().add_sampler(name, source, config_file, test_file=test_file)
        print("Added importance sampler {}".format(name))

    @classmethod
    def remove_sampler(self, name=None):
        ImportanceSamplerManager().remove_sampler(name)
        print("Removed importance sampler {}".format(name))

    @classmethod
    def test_sampler(cls, name=None):
        print("Testing importance sampler {}".format(name))
        ke = ImportanceSamplerManager().test_sampler(name)
        print("Sampler returned local kinetic energy {}".format(ke))

    @classmethod
    def test_ch5_sampler(cls):
        im = ImportanceSamplerManager()
        if "CH5" in im.list_samplers():
            im.remove_sampler("CH5")
        im.add_sampler(
            "CH5",
            os.path.join(os.path.dirname(__file__), "Tests", "TestData", "CH5TrialWavefunction"),
            config_file=os.path.join(os.path.dirname(__file__), "Tests", "TestData", "ch5_sampler.py")
        )

class PotentialInterface:
    """
    Defines all of the CLI options for working with potentials
    """

    @classmethod
    def list_potentials(self):
        print("\n".join(PotentialManager().list_potentials()))

    @classmethod
    def add_potential(self, name=None, src=None, config_file=None, data=None, test_file=None):
        no_config = config_file is None
        if config_file is None:
            if os.path.exists(os.path.join(src, "config.py")):
                config_file = os.path.join(src, "config.py")
        if data is None:
            if os.path.exists(os.path.join(src, "data")):
                data = os.path.join(src, "data")
        if test_file is None:
            if os.path.exists(os.path.join(src, "test.py")):
                test_file = os.path.join(src, "test.py")
        if no_config:
            if os.path.exists(os.path.join(src, name)):
                src = os.path.join(src, name)
        PotentialManager().add_potential(name, src, config_file=config_file, data=data, test=test_file)
        print("Added potential {}".format(name))

    @classmethod
    def remove_potential(self, name=None):
        PotentialManager().remove_potential(name)
        print("Removed potential {}".format(name))

    @classmethod
    def export_potential(cls, name=None, path=None):
        PotentialManager().export_potential(name, path)

    @classmethod
    def compile_potential(self, name=None):
        PotentialManager().compile_potential(name)

    @classmethod
    def configure_entos(cls):
        pm = PotentialManager()
        entos = RynLib.get_conf().entos_binary
        pm.add_potential(
            "entos",
            src=entos,
            wrap_potential=True,
            function_name="MillerGroup_entosPotential",
            arguments=(("only_hf", 'bool'),),
            linked_libs=["entos"],
            include_dirs=[os.path.dirname(entos)],
            static_source = True
        )
        # writes to /config/potentials/entos/src/entos.cpp
        pm.compile_potential('entos')

    @classmethod
    def configure_HO(cls):
        pm = PotentialManager()
        pm.add_potential(
            "HarmonicOscillator",
            src=os.path.join(os.path.dirname(__file__), "Tests", "TestData", "HarmonicOscillator"),
            wrap_potential=True,
            function_name='HarmonicOscillator',
            arguments=(('re', 'float'), ('k', 'float')),
            requires_make=True,
            linked_libs=['HarmonicOscillator']
        )
        pm.compile_potential('HarmonicOscillator')

    @classmethod
    def potential_status(self, name=None):
        pm = PotentialManager()
        status = pm.potential_compiled(name)
        config = pm.potential_config(name)
        print(
            "Compiled: {}".format(status),
            *("  {}: {}".format(k, v) for k, v in config.opt_dict.items()),
            sep="\n"
        )

    @classmethod
    def test_potential(self, name=None, input_file=None):

        print("Testing Potential: {}".format(name))
        print("Energies: {}".format(PotentialManager().test_potential(name), input_file=input_file))

    @classmethod
    def test_potential_mpi(self, name=None, input_file=None, **opts):

        PotentialManager().test_potential_mpi(name, input_file=input_file, **opts)

class RynLib:
    """
    Defines all of the overall RynLib config things
    """
    config_file = "config.py"
    @classmethod
    def get_default_env(cls):
        import platform
        node = platform.node()
        if 'hyak' in node:
            env = dict(
                containerizer="singularity",
                entos_binary="/entos/lib/libentos.so",
                root_directory="",
                simulation_directory="./simulations",
                sampler_directory="./impsamps",
                potential_directory="./potentials",
                mpi_version="3.1.4",
                mpi_implementation="ompi",
                mpi_dir="./libs",
                mpi_flags=[
                    "--disable-oshmem",
                    "--enable-branch-probabilities",
                    "--disable-mpi-fortran",
                    "--with-slurm"
                    #,
                    # "--with-pmi=/usr",
                    # "--with-psm2=/usr"
                ]
            )
        elif 'cori' in node:
            env = dict(
                containerizer="shifter",
                entos_binary="/entos/lib/libentos.so",
                root_directory="/config",
                simulation_directory="/config/simulations",
                sampler_directory="/config/impsamps",
                potential_directory="/config/potentials",
                mpi_version="3.2",
                mpi_implementation="mpich",
                mpi_dir="/config/mpi",
                mpi_flags=[
                    "--disable-oshmem",
                    "--enable-branch-probabilities",
                    "--disable-fortran",
                    "--disable-mpi-fortran"
                ]
            )
        else:
            env = dict(
                containerizer="docker",
                entos_binary="/entos/lib/libentos.so",
                root_directory="/config",
                simulation_directory="/config/simulations",
                sampler_directory="/config/impsamps",
                potential_directory="/config/potentials",
                mpi_version="3.1.4",
                mpi_implementation="ompi",
                mpi_dir="/config/mpi",
                mpi_flags=[]
            )
        return env

    @classmethod
    def get_conf(cls):
        default_env = cls.get_default_env()
        conf_path = os.path.join(default_env["root_directory"], cls.config_file)
        new_conf = not os.path.exists(conf_path)
        if new_conf:
            ConfigSerializer.serialize(
                conf_path,
                default_env,
                attribute="config"
            )
        cfig = Config(conf_path)
        return cfig

    @classmethod
    def reset_config(cls):
        default_env = cls.get_default_env()
        conf_path = os.path.join(default_env["root_directory"], cls.config_file)
        ConfigSerializer.serialize(
            conf_path,
            default_env,
            attribute="config"
        )
    @classmethod
    def edit_config(cls, **opts):
        op2 = {k:v for k,v in opts.items() if (v is not None and not (isinstance(v, str) and v==""))}
        cls.get_conf().update(**op2)
    @classmethod
    def get_container_env(cls):
        return cls.get_conf().containerizer

    @classmethod
    def build_libs(cls):
        from .PlzNumbers import PotentialCaller
        PotentialCaller._load_lib()

        # if os.path.isdir(os.path.join(cls.get_conf().mpi_dir)):
        #     cls.reload_dumpi()


    @classmethod
    def run_tests(cls):

        curdir = os.getcwd()
        try:
            os.chdir(os.path.dirname(os.path.dirname(__file__)))
            import RynLib.Tests.run_tests
        finally:
            os.chdir(curdir)

    @classmethod
    def update_lib(cls):
        """
        Pulls the updated RynLib from GitHub

        :return:
        :rtype:
        """
        import subprocess

        curdir = os.getcwd()
        try:
            os.chdir(os.path.dirname(__file__))
            print(subprocess.check_call(["git", "pull"]))
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise
        finally:
            os.chdir(curdir)

    @classmethod
    def update_testing_framework(cls):
        """
        Pulls the updated RynLib from GitHub

        :return:
        :rtype:
        """
        import subprocess

        curdir = os.getcwd()
        try:
            os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Peeves"))
            print(subprocess.check_call(["git", "pull"]))
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise
        finally:
            os.chdir(curdir)

    @classmethod
    def install_MPI(cls):
        """Installs MPI into the containerized environment"""
        #This comes from /sw/singularity-images/testing/ngsolve-2.def
        import subprocess, tempfile, wget, tarfile

        conf = cls.get_conf()

        MPI_DIR = os.path.join(os.path.abspath(conf.mpi_dir), "mpi")
        MPI_IMP = conf.mpi_implementation.lower()

        if os.path.isdir(MPI_DIR):
            shutil.rmtree(MPI_DIR)
        os.makedirs(MPI_DIR)

        MPI_VERSION = conf.mpi_version
        MPI_MAJOR_VERSION = ".".join(MPI_VERSION.split(".")[:2])
        if MPI_IMP == "ompi":
            MPI_URL = "https://download.open-mpi.org/release/open-mpi/v{MPI_MAJOR_VERSION}/openmpi-{MPI_VERSION}.tar.bz2".format(
                MPI_MAJOR_VERSION = MPI_MAJOR_VERSION,
                MPI_VERSION = MPI_VERSION
            )
        else:
            MPI_URL = "https://www.mpich.org/static/downloads/{MPI_VERSION}/mpich-{MPI_VERSION}.tar.gz"

        with tempfile.TemporaryDirectory() as build_dir:
            if MPI_IMP == "ompi":
                mpi_ext = "openmpi-{MPI_VERSION}".format(MPI_VERSION=MPI_VERSION)
                mpi_src = os.path.join(build_dir, mpi_ext + ".tar.bz2")
            elif MPI_IMP == "mpich":
                mpi_ext = "mpich-{MPI_VERSION}".format(MPI_VERSION=MPI_VERSION)
                mpi_src = os.path.join(build_dir, mpi_ext + ".tar.gz")

            print("Downloading MPI from {}".format(MPI_URL))

            wget.download(MPI_URL, mpi_src)
            with tarfile.open(mpi_src) as tar:
                tar.extractall(build_dir)

            curdir = os.getcwd()
            try:
                os.chdir(os.path.join(build_dir, mpi_ext))
                print("\nCompiling MPI")
                subprocess.check_output([
                    "./configure",
                    "--prefix={MPI_DIR}".format(MPI_DIR=MPI_DIR),
                    *conf.mpi_flags
                ])
                subprocess.check_output([
                    "make",
                    "-j4",
                    "install"
                ])
                subprocess.check_output([
                    "make",
                    "clean"
                ])
            except subprocess.CalledProcessError as e:
                print(e.output.decode())
                raise
            finally:
                os.chdir(curdir)

        print("\nInstalling MPI to {}",format(MPI_DIR))

        conf.update(mpi_dir=MPI_DIR)

    @classmethod
    def configure_mpi(cls):
        mpi_dir = cls.get_conf().mpi_dir
        if not os.path.isdir(mpi_dir):
            cls.install_MPI()

        from .Dumpi import MPIManagerObject
        MPIManagerObject._load_lib()

    @classmethod
    def reload_dumpi(cls):
        mpi_dir = cls.get_conf().mpi_dir
        if not os.path.isdir(mpi_dir):
            cls.install_MPI()

        from .Dumpi import MPIManagerObject
        try:
            MPIManagerObject._remove_lib()
        except OSError:
            pass
        MPIManagerObject._load_lib()

    @classmethod
    def test_mpi(cls):
        mpi_dir = cls.get_conf().mpi_dir
        if not os.path.isdir(mpi_dir):
            cls.install_MPI()

        from .Dumpi import MPIManagerObject
        manager = MPIManagerObject()
        print("World Size: {} World Rank: {}".format(manager.world_size, manager.world_rank))

    @classmethod
    def test_entos(cls):
        import numpy as np

        testWalker = np.array([
            [0.9578400, 0.0000000, 0.0000000],
            [-0.2399535, 0.9272970, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000]
        ])
        testAtoms = ["H", "H", "O"]

        potential_manager = PotentialManager()
        if 'entos' not in potential_manager.list_potentials():
            PotentialInterface.configure_entos()

        entos = PotentialManager().load_potential("entos")

        print("Testing Entos:")

        print("Energy w/ MOBML: {}".format(entos(testWalker, testAtoms, False)))
        print("Energy w/o MOBML: {}".format(entos(testWalker, testAtoms, True)))

    @classmethod
    def test_HO(cls):

        potential_manager = PotentialManager()
        if 'HarmonicOscillator' not in potential_manager.list_potentials():
            PotentialInterface.configure_HO()
        HO = PotentialManager().load_potential("HarmonicOscillator")

        print("Testing Harmonic Oscillator:")
        print("Energy of HO: {}".format(HO([[1, 2, 3], [1, 1, 1]], ["H", "H"], .9, 1.)))

    @classmethod
    def test_potential_mpi(cls, # duplicated for now, but oh well
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
        from .Dumpi import MPIManager, MPIManagerObject

        mpi_manager = MPIManager()

        if mpi_manager is None:
            raise ImportError("MPI isn't installed. Use `container config install_mpi` first.")

        mpi = mpi_manager #type: MPIManagerObject

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
        testWalkersss = np.broadcast_to(testWalkersss, (nsteps,) + testWalkersss.shape)

        #
        # run tests
        #
        potential.mpi_manager = mpi_manager
        test_results_for_real = np.zeros((test_iterations, nsteps, num_walkers))
        for ttt in range(test_iterations):
            t0 = time.time()
            # call the potential
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
            print("Average total: {}s Average time per walker: {}s".format(np.average(test_results), np.average(
                test_results) / num_walkers / nsteps))

            mpi_manager.finalize_MPI()

    @classmethod
    def test_entos_mpi(cls,
                       walkers_per_core=5,
                       displacement_radius=.5,
                       iterations=5,
                       steps_per_call=5,
                       print_walkers=False
                       ):
        import numpy as np, time
        from .Dumpi import MPIManager, MPIManagerObject

        potential_manager = PotentialManager()
        if 'entos' not in potential_manager.list_potentials():
            PotentialInterface.configure_entos()

        entos = PotentialManager().load_potential("entos")

        testWalker = np.array([
            [0.9578400, 0.0000000, 0.0000000],
            [-0.2399535, 0.9272970, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000]
        ])
        testAtoms = ["H", "H", "O"]

        cls.test_potential_mpi(
            entos,
            testWalker,
            testAtoms,
            False,
            walkers_per_core=walkers_per_core,
            displacement_radius=displacement_radius,
            iterations=iterations,
            steps_per_call=steps_per_call,
            print_walkers=print_walkers
        )

    @classmethod
    def test_ho_mpi(cls,
                       walkers_per_core=5,
                       displacement_radius=.5,
                       iterations=5,
                       steps_per_call=5,
                       print_walkers=False
                       ):
        import numpy as np

        potential_manager = PotentialManager()
        if 'HarmonicOscillator' not in potential_manager.list_potentials():
            PotentialInterface.configure_HO()

        ho = PotentialManager().load_potential("HarmonicOscillator")

        testWalker = np.array([[1, 2, 3], [1, 1, 1]])
        testAtoms = ["H", "H"]

        cls.test_potential_mpi(
            ho,
            testWalker,
            testAtoms,
            .9,
            1.,
            walkers_per_core=walkers_per_core,
            displacement_radius=displacement_radius,
            iterations=iterations,
            steps_per_call=steps_per_call,
            print_walkers=print_walkers
        )

class ContainerException(IOError):
    ...