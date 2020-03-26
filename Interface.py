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
    "GeneralConfig"
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
    def add_potential(self, name=None, src=src, config_file=config_file):
        PotentialManager().add_potential(name, config_file, src)
        print("Added potential {}".format(name))

    @classmethod
    def remove_potential(self, name=None):
        PotentialManager().remove_potential(name)
        print("Removed potential {}".format(name))

    @classmethod
    def compile_potential(self, name=None):
        PotentialManager().compile_potential(name)

class GeneralConfig:
    """
    Defines all of the overall RynLib config things
    """
    config_file = "config.py"
    root = os.path.dirname(__file__)
    @classmethod
    def get_conf(cls):
        conf_path = os.path.join(cls.root, cls.config_file)
        new_conf = not os.path.exists(conf_path)
        if new_conf:
            ConfigSerializer.serialize(
                conf_path,
                dict(
                    containerizer="singularity",
                    simulation_directory="./simulations",
                    potential_directory="./potentials",
                    entos_binary="...",
                    mpi_version="3.1.4",
                    mpi_dir="/opt/ompi"
                ),
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

        OMPI_DIR = conf.mpi_dir

        if os.path.isdir(OMPI_DIR):
            shutil.rmtree(OMPI_DIR)

        OMPI_VERSION = conf.mpi_version
        OMPI_MAJOR_VERSION = ".".join(OMPI_VERSION.split(".")[:2])
        OMPI_URL = "https://download.open-mpi.org/release/open-mpi/v{OMPI_MAJOR_VERSION}/openmpi-{OMPI_VERSION}.tar.bz2".format(
            OMPI_MAJOR_VERSION = OMPI_MAJOR_VERSION,
            OMPI_VERSION = OMPI_VERSION
        )

        with tempfile.TemporaryDirectory() as build_dir:
            ompi_ext = "openmpi-{OMPI_VERSION}".format(OMPI_VERSION=OMPI_VERSION)
            mpi_src = os.path.join(build_dir, ompi_ext+".tar.bz2")
            wget.download(OMPI_URL, mpi_src)
            with tarfile.open(mpi_src) as tar:
                tar.extractall(build_dir)
            curdir = os.getcwd()
            try:
                os.chdir(os.path.join(ompi_ext))
                print(subprocess.check_output([
                    "./configure",
                    "--prefix={OMPI_DIR}".format(OMPI_DIR=OMPI_DIR),
                    "--disable-oshmem",
                    "--enable-branch-probabilities"
                ]))
                print(subprocess.check_output([
                    "make",
                    "-j12",
                    "install"
                ]))
            except subprocess.CalledProcessError as e:
                print(e.output)
                raise
            finally:
                os.chdir(curdir)

        conf.update(mpi_dir=OMPI_DIR)

    @classmethod
    def create_volumes(cls):
        """
        Binds the appropriate volumes for read/write
            -> currently this just checks to see if the read/write mounts will work...

        :return:
        :rtype:
        """
        config = cls.get_conf()
        sim_dir = config.simulation_directory
        pot_dir = config.potential_directory
        containerizer = config.containerizer.lower()
        if containerizer == "singularity":
            known_paths = ["~", ".", "/proc", "/tmp"]
            sim_root = os.path.dirname(sim_dir)
            if sim_root not in known_paths:
                raise ContainerException("Haven't implemented handling bind paths for singularity")
            pot_root = os.path.dirname(pot_dir)
            if pot_root not in known_paths:
                raise ContainerException("Haven't implemented handling bind paths for singularity")
        elif containerizer == "docker":
            # this might not work in general...
            known_paths = ["~", ".", "/proc", "/tmp"]
            sim_root = os.path.dirname(sim_dir)
            if sim_root not in known_paths:
                raise ContainerException("Haven't implemented handling bind paths for docker")
            pot_root = os.path.dirname(pot_dir)
            if pot_root not in known_paths:
                raise ContainerException("Haven't implemented handling bind paths for docker")
        else:
            # we assume this is local
            pass

class ContainerException(IOError):
    ...