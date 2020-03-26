"""
The overall interface to the RynLib CLI
"""

import os, shutil
from .RynUtils import Config, ConfigSerializer
from .PlzNumbers import PotentialManager
from .DoMyCode import SimulationManager

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
    def run_simulation(self, name=None):
        SimulationManager().run_simulation(name)

    @classmethod
    def restart_simulation(self, name=None):
        SimulationManager().restart_simulation(name)

class PotentialInterface:
    """
    Defines all of the CLI options for working with potentials
    """

    @classmethod
    def list_potentials(self):
        print("\n".join(PotentialManager().list_potentials()))

    @classmethod
    def add_potential(self, name=None, src=None, config_file=None):
        PotentialManager().add_potential(name, config_file, src)
        print("Added potential {}".format(name))

    @classmethod
    def remove_potential(self, name=None):
        PotentialManager().remove_potential(name)
        print("Removed potential {}".format(name))

    @classmethod
    def compile_potential(self, name=None):
        PotentialManager().compile_potential(name)

    @classmethod
    def configure_entos(cls):
        PotentialManager().add_potential(
            "entos",
            src=RynLib.get_conf().entos_binary,
            wrap_potential=True,
            function_name="MillerGroup_entosPotential",
            arguments=(("only_hf", bool),)
        )

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
                root_directory="",
                simulation_directory="simulations",
                potential_directory="potentials",
                mpi_version="3.1.4",
                mpi_implementation="ompi",
                mpi_dir="/proc/mpi"
            )
        elif 'cori' in node:
            env = dict(
                containerizer="shifter",
                root_directory="/config",
                simulation_directory="/config/simulations",
                potential_directory="/config/potentials",
                mpi_version="3.2",
                mpi_implementation="mpich",
                mpi_dir="/config/mpi"
            )
        else:
            env = dict(
                containerizer="docker",
                root_directory="/config",
                simulation_directory="/config/simulations",
                potential_directory="/config/potentials",
                mpi_version="3.1.4",
                mpi_implementation="ompi",
                mpi_dir="/config/mpi"
            )
        return env
    @classmethod
    def get_conf(cls):
        default_env = cls.get_container_env()
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
    def edit_config(cls, **opts):
        op2 = {k:v for k,v in opts.items() if (v is not None and not (isinstance(v, str) and v==""))}
        cls.get_conf().update(**op2)
    @classmethod
    def get_container_env(cls):
        return cls.get_conf().containerizer

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

        MPI_DIR = conf.mpi_dir
        MPI_IMP = conf.mpi_implementation.lower()

        if os.path.isdir(MPI_DIR):
            shutil.rmtree(MPI_DIR)

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
            wget.download(MPI_URL, mpi_src)
            with tarfile.open(mpi_src) as tar:
                tar.extractall(build_dir)

            curdir = os.getcwd()
            try:
                os.chdir(os.path.join(mpi_ext))
                print(subprocess.check_output([
                    "./configure",
                    "--prefix={MPI_DIR}".format(MPI_DIR=MPI_DIR),
                    "--disable-oshmem",
                    "--enable-branch-probabilities"
                ]))
                print(subprocess.check_output([
                    "make",
                    "-j4",
                    "install"
                ]))
                print(subprocess.check_output([
                    "make",
                    "clean"
                ]))
            except subprocess.CalledProcessError as e:
                print(e.output)
                raise
            finally:
                os.chdir(curdir)

        conf.update(mpi_dir=MPI_DIR)

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

        testWalker = np.array([
            [0.9578400, 0.0000000, 0.0000000],
            [-0.2399535, 0.9272970, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000]
        ])
        testAtoms = ["H", "H", "O"]

        mpi_manager = MPIManager()

        if mpi_manager is None:
            raise ImportError("MPI isn't installed. Use `container config install_mpi` first.")

        mpi = MPIManagerObject()

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
        testWalkersss = np.array([testWalker] * num_walkers)
        testWalkersss += np.random.uniform(low=-displacement_radius, high=displacement_radius,
                                           size=testWalkersss.shape)
        test_iterations = iterations
        test_results = np.zeros((test_iterations,))
        lets_get_going = time.time()
        nsteps = steps_per_call
        # we compute the same walker for each of the nsteps, but that's okay -- gives a nice clean test that everything went right
        testWalkersss = np.broadcast_to(testWalkersss, (nsteps,) + testWalkersss.shape)

        potential_manager = PotentialManager()
        if 'entos' not in potential_manager.list_potentials():
            PotentialInterface.configure_entos()

        entos = PotentialManager().load_potential("entos")

        #
        # run tests
        #
        test_results_for_real = np.zeros((test_iterations, nsteps, num_walkers))
        for ttt in range(test_iterations):
            t0 = time.time()
            # this gets the correct memory layout
            test_walkers_reshaped = np.ascontiguousarray(testWalkersss.transpose(1, 0, 2, 3))
            # call the potential
            test_result = entos(
                testAtoms,
                test_walkers_reshaped,
                True# hf_only argument is necessary with current setup
            )
            # then we gotta transpose back to the input layout
            if who_am_i == 0:
                test_results_for_real[ttt] = test_result.transpose()
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
                    "Got back result with shape {}".format(test_result.shape),
                    sep="\n"
                )
            print("Total time: {}s (over {} iterations)".format(gotta_go_fast, test_iterations))
            print("Average total: {}s Average time per walker: {}s".format(np.average(test_results), np.average(
                test_results) / num_walkers / nsteps))

class ContainerException(IOError):
    ...