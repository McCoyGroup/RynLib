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
    def add_simulation(self, name=None, src=None, config_file=None):
        data=src
        no_config = config_file is None
        if config_file is None:
            if os.path.exists(os.path.join(data, "config.py")):
                config_file = os.path.join(data, "config.py")
        if no_config:
            if os.path.exists(os.path.join(data, "data")):
                data = os.path.join(data, "data")
        SimulationManager().add_simulation(name, data=data, config_file=config_file)
        print("Added simulation {}".format(name))

    @classmethod
    def remove_simulation(self, name=None):
        SimulationManager().remove_simulation(name)
        print("Removed simulation {}".format(name))

    @classmethod
    def copy_simulation(self, name=None, new_name=None):
        SimulationManager().copy_simulation(name, new_name)
        print("Copied simulation {} into {}".format(name, new_name))

    @classmethod
    def simulation_status(self, name=None):
        status = SimulationManager().simulation_ran(name)
        config = SimulationManager().simulation_config(name)
        print(
            "Has Run: {}".format(status),
            *("  {}: {}".format(k, v) for k, v in config.opt_dict.items()),
            sep="\n"
        )

    @classmethod
    def edit_simulation(self, name=None, opts=None, optfile=None):
        SimulationManager().edit_simulation(name, optfile=optfile, **opts)
        print("Edited simulation {}".format(
            name
        ))
        self.simulation_status(name)

    @classmethod
    def export_simulation(cls, name=None, path=None):
        SimulationManager().export_simulation(name, path)
        print("Exported simulation {} to {}".format(
            name, path
        ))

    @classmethod
    def run_simulation(self, name=None):
        # print("Running simulation {}".format(name))
        SimulationManager().run_simulation(name)
        # print("Finished running simulation {}".format(name))

    @classmethod
    def restart_simulation(self, name=None):
        # print("Running simulation {}".format(name))
        SimulationManager().restart_simulation(name)
        # print("Finished running simulation {}".format(name))

    @classmethod
    def test_add_HO(cls):
        pm = PotentialManager()
        if 'HarmonicOscillator' not in pm.list_potentials():
            PotentialInterface.configure_HO()
        sm = SimulationManager()
        if "test_HO" in sm.list_simulations():
            sm.remove_simulation("test_HO")
        cls.add_simulation("test_HO",
                           os.path.join(RynLib.test_data, "HOSimulation", "HOSim")
                           )

    @classmethod
    def test_HO(cls):
        cls.test_add_HO()
        cls.run_simulation("test_HO")

    @classmethod
    def test_HO_imp(cls):
        im = ImportanceSamplerManager()
        if "HOSampler" not in im.list_samplers():
            SimulationInterface.add_sampler(
                "HOSampler",
                source=os.path.join(RynLib.test_data, "HOSimulation", "HOTrialWavefunction")
            )
        sm = SimulationManager()
        if "test_HO_imp" in sm.list_simulations():
            sm.remove_simulation("test_HO_imp")
        pm = PotentialManager()
        if 'HarmonicOscillator' not in pm.list_potentials():
            PotentialInterface.configure_HO()
        cls.add_simulation("test_HO_imp",
                           os.path.join(RynLib.test_data, "HOSimulation", "HOSimImp")
                           )
        cls.run_simulation("test_HO_imp")

    @classmethod
    def list_archive(cls):
        print("\n".join(SimulationManager().list_archive()))

    @classmethod
    def archive_simulation(cls, name=None):
        SimulationManager().archive_simulation(name)
        print("Archived simulation {}".format(name))

    @classmethod
    def archive_status(cls, name=None):
        config = SimulationManager().archive_config(name)
        print(
            *("{}: {}".format(k, v) for k, v in config.opt_dict.items()),
            sep="\n"
        )

    @classmethod
    def list_samplers(cls):
        print("\n".join(ImportanceSamplerManager().list_samplers()))

    @classmethod
    def add_sampler(self, name=None, source=None, config_file=None, test_file=None):
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
        import numpy as np

        print("Testing importance sampler {}".format(name))
        ke, meta = ImportanceSamplerManager().test_sampler(name)
        print("Sampler returned average local kinetic energy {}".format(np.average(ke, axis=1)))

    @classmethod
    def test_sampler_mpi(cls, name=None, input_file=None, **opts):
        import numpy as np

        samp = ImportanceSamplerManager().test_sampler_mpi(name, input_file=input_file, **opts)
        mpi = next(samp)
        if mpi.world_rank == 0:
            print("Testing importance sampler {}".format(name))
            print(mpi)
        ke, meta = next(samp)
        mpi.finalize_MPI()
        if mpi.world_rank == 0:
            print(meta['walkers'])
            print("Took {}s ({}s/walker)".format(meta['timing'], meta['average'])),
            print("Sampler returned average local kinetic energy {}".format(np.average(ke, axis=1)))
        mpi.finalize_MPI()

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
    def import_potential(cls, name=None, src=None):
        PotentialManager().import_potential(name, src)

    @classmethod
    def export_potential(cls, name=None, dest=None):
        PotentialManager().export_potential(name, dest)

    @classmethod
    def compile_potential(self, name=None, recompile=False):
        PotentialManager().compile_potential(name, recompile=recompile)

    @classmethod
    def configure_entos(cls):
        pm = PotentialManager()
        entos = RynLib.get_conf().entos_binary
        pm.add_potential(
            "entos",
            src=entos,
            test=os.path.join(RynLib.test_data, "test_entos.py"),
            wrap_potential=True,
            function_name="MillerGroup_entosPotential",
            working_directory="/opt/entos",
            arguments=(("only_hf", 'bool'),),
            linked_libs=["entos"],
            include_dirs=[os.path.dirname(entos)],
            static_source = True
        )
        # writes to /config/potentials/entos/src/entos.cpp
        pm.compile_potential('entos')

    @classmethod
    def configure_HO(cls):
        cls.add_potential(
            "HarmonicOscillator",
            os.path.join(RynLib.test_data, "HOSimulation", "HOPotential")
        )
        PotentialManager().compile_potential('HarmonicOscillator')

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
        print("Energies: {}".format(PotentialManager().test_potential(name, input_file=input_file)))

    @classmethod
    def test_potential_mpi(self, name=None, input_file=None, **opts):
        PotentialManager().test_potential_mpi(name, input_file=input_file, **opts)

    @classmethod
    def test_potential_serial(self, name=None, input_file=None, **opts):
        PotentialManager().test_potential_serial(name, input_file=input_file, **opts)

class RynLib:
    """
    Defines all of the overall RynLib config things
    """
    from RynLib import VERSION_NUMBER
    VERSION_NUMBER = VERSION_NUMBER
    root = "/config"
    config_file = "config.py"
    mpi_dir = "/usr/lib/mpi"
    test_data = os.path.join(os.path.dirname(__file__), "Tests", "TestData")

    flags = dict(
        multiprocessing = False,
        OpenMPThreads = True,
        TBBThreads = False
    )

    @classmethod
    def get_default_env(cls):
        import platform
        node = platform.node()
        if 'hyak' in node:
            env = dict(
                containerizer="singularity",
                entos_binary="/opt/entos/lib/libentos.so",
                root_directory="#",
                simulation_directory="#/simulations",
                sampler_directory="#/impsamps",
                potential_directory="#/potentials"
            )
        elif 'cori' in node:
            env = dict(
                containerizer="shifter",
                entos_binary="/opt/entos/lib/libentos.so",
                root_directory="#",
                simulation_directory="#/simulations",
                sampler_directory="#/impsamps",
                potential_directory="#/potentials"
            )
        else:
            env = dict(
                containerizer="docker",
                entos_binary="/opt/entos/lib/libentos.so",
                root_directory="#",
                simulation_directory="#/simulations",
                sampler_directory="#/impsamps",
                potential_directory="#/potentials"
            )
        return env

    @classmethod
    def get_conf(cls):
        conf_path = os.path.join(cls.root, cls.config_file)
        new_conf = not os.path.exists(conf_path)
        if new_conf:
            ConfigSerializer.serialize(
                conf_path,
                cls.get_default_env(),
                attribute="config"
            )
        cfig = Config(conf_path)
        return cfig

    @classmethod
    def root_directory(cls):
        cf = cls.get_conf()
        return cf.root_directory.replace("#", cls.root)

    @classmethod
    def simulation_directory(cls):
        cf = cls.get_conf()
        sim_dir = cf.simulation_directory
        return os.path.abspath(sim_dir.replace("#", cls.root_directory()))

    @classmethod
    def potential_directory(cls):
        cf = cls.get_conf()
        pot_dir = cf.potential_directory
        return os.path.abspath(pot_dir.replace("#", cls.root_directory()))

    @classmethod
    def sampler_directory(cls):
        cf = cls.get_conf()
        samp_dir = cf.sampler_directory
        return os.path.abspath(samp_dir.replace("#", cls.root_directory()))

    @classmethod
    def reset_config(cls):
        default_env = cls.get_default_env()
        conf_path = os.path.join(cls.root, cls.config_file)
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
    def build_libs(cls, rebuild=False):
        from .PlzNumbers import PotentialCaller
        from .DoMyCode import Simulation
        if rebuild:
            PotentialCaller.reload()
            Simulation.reload_lib()
            cls.reload_dumpi()
        else:
            Simulation.load_lib()
            PotentialCaller.load_lib()
            cls.configure_mpi()

        # if os.path.isdir(os.path.join(cls.get_conf().mpi_dir)):
        #     cls.reload_dumpi()

    @classmethod
    def run_tests(cls, test_dir=None, debug=False, name=None, suite=None):
        import sys

        curdir = os.getcwd()
        root = cls.root
        argv = sys.argv
        if test_dir is None:
            no_user_dir = True
            test_dir = "/tests"
        else:
            no_user_dir = False
        try:
            cls.root = test_dir
            if no_user_dir and os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            os.chdir(os.path.dirname(os.path.dirname(__file__)))
            if debug:

                sys.argv = [argv[0], "-d"]
            else:
                sys.argv = [argv[0], "-v", "-d"]

            if name is not None:
                sys.argv.extend(["-n",  name])
            if suite is not None:
                sys.argv.extend(["-f", suite])

            import RynLib.Tests.run_tests
        finally:
            sys.argv = argv
            cls.root = root
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
            subprocess.call(["git", "pull"])#print()
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
    def install_MPI(cls,
                    mpi_version=None,
                    mpi_implementation=None,
                    mpi_dir=None,
                    mpi_flags=None,
                    bind_conf = False
                    ):
        """Installs MPI into the containerized environment"""
        #This comes from /sw/singularity-images/testing/ngsolve-2.def
        import subprocess, tempfile, wget, tarfile

        conf = None
        if mpi_dir is None:
            mpi_dir=cls.mpi_dir
        if mpi_implementation is None:
            if conf is None:
                conf = cls.get_conf()
            mpi_implementation=conf.mpi_implementation
        if mpi_version is None:
            if conf is None:
                conf = cls.get_conf()
            mpi_version=conf.mpi_version
        if mpi_flags is None:
            mpi_flags = (
                "--disable-oshmem",
                "--enable-branch-probabilities",
                "--disable-mpi-fortran"
            )
            # if conf is None:
            #     conf = cls.get_conf()
            # mpi_flags = conf.mpi_flags

        MPI_DIR = os.path.abspath(mpi_dir) # os.path.join(mpi_dir, "mpi")
        MPI_IMP = mpi_implementation.lower()

        if os.path.isdir(MPI_DIR):
            shutil.rmtree(MPI_DIR)
        os.makedirs(MPI_DIR)

        MPI_VERSION = mpi_version
        MPI_MAJOR_VERSION = ".".join(MPI_VERSION.split(".")[:2])
        if MPI_IMP == "ompi":
            MPI_URL = "https://download.open-mpi.org/release/open-mpi/v{MPI_MAJOR_VERSION}/openmpi-{MPI_VERSION}.tar.bz2".format(
                MPI_MAJOR_VERSION = MPI_MAJOR_VERSION,
                MPI_VERSION = MPI_VERSION
            )
        else:
            MPI_URL = "https://www.mpich.org/static/downloads/{MPI_VERSION}/mpich-{MPI_VERSION}.tar.gz".format(
                MPI_MAJOR_VERSION = MPI_MAJOR_VERSION,
                MPI_VERSION = MPI_VERSION
            )

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
                    *mpi_flags
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

        if bind_conf:
            conf.update(mpi_dir=MPI_DIR)

    @classmethod
    def configure_mpi(cls):
        mpi_dir = cls.mpi_dir#cls.mpi_dir#cls.get_conf().mpi_dir
        if not os.path.isdir(mpi_dir):
            cls.install_MPI()

        from .Dumpi import MPIManagerObject
        MPIManagerObject._load_lib()

    @classmethod
    def reload_dumpi(cls):
        mpi_dir = cls.mpi_dir#cls.get_conf().mpi_dir
        if not os.path.isdir(mpi_dir):
            cls.install_MPI()

        from .Dumpi import MPIManagerObject
        # try:
        MPIManagerObject._remove_lib()
        # except OSError:
        #     pass
        MPIManagerObject._load_lib()

    @classmethod
    def test_mpi(cls):
        mpi_dir = cls.mpi_dir#cls.get_conf().mpi_dir
        if not os.path.isdir(mpi_dir):
            cls.install_MPI()

        from .Dumpi import MPIManagerObject
        manager = MPIManagerObject()
        print("World Size: {} World Rank: {}".format(manager.world_size, manager.world_rank))

    @classmethod
    def test_entos(cls):

        potential_manager = PotentialManager()
        if 'entos' not in potential_manager.list_potentials():
            PotentialInterface.configure_entos()

        print("Testing Entos:")
        # try:
        #     val = potential_manager.test_potential('entos')
        # except Exception as E:
        #     val = E.args[0]
        # print("Energy (no hf_only flag): {}".format(val))
        print("Energy w/ MOBML: {}".format(potential_manager.test_potential('entos', parameters=dict(only_hf=False))))
        print("Energy w/o MOBML: {}".format(potential_manager.test_potential('entos', parameters=dict(only_hf=True))))

    @classmethod
    def test_HO(cls):

        potential_manager = PotentialManager()
        if 'HarmonicOscillator' not in potential_manager.list_potentials():
            PotentialInterface.configure_HO()

        print("Testing Harmonic Oscillator:")
        print("Energy of HO: {}".format(potential_manager.test_potential("HarmonicOscillator")))

    @classmethod
    def test_entos_mpi(cls,
                       **opts
                       ):

        potential_manager = PotentialManager()
        if 'entos' not in potential_manager.list_potentials():
            PotentialInterface.configure_entos()

        potential_manager.test_potential_mpi(
            "entos",
            **opts
        )

    @classmethod
    def test_ho_mpi(cls, **opts):

        potential_manager = PotentialManager()
        if 'HarmonicOscillator' not in potential_manager.list_potentials():
            PotentialInterface.configure_HO()

        potential_manager.test_potential_mpi(
            "HarmonicOscillator",
            **opts
        )

class ContainerException(IOError):
    ...