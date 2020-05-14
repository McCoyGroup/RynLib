from ..RynUtils import ConfigManager
from .Simulation import Simulation, SimulationParameters
import os, shutil

__all__ = [
    "SimulationManager"
]

class SimulationManager:
    archive_path = "archives"
    def __init__(self, config_dir=None):
        if config_dir is None:
            from ..Interface import RynLib
            config_dir = RynLib.simulation_directory()
        self.manager = ConfigManager(config_dir)
        self.archiver = ConfigManager(os.path.join(config_dir, self.archive_path))

    def check_simulation(self, name):
        if not self.manager.check_config(name):
            raise IOError("No simulation {}".format(name))

    def list_simulations(self):
        return self.manager.list_configs()

    def remove_simulation(self, name):
        self.check_simulation(name)
        self.manager.remove_config(name)

    def add_simulation(self, name, data=None, config_file = None, **opts):
        self.manager.add_config(name, config_file = config_file, **opts)
        if data is not None:
            data_src = os.path.join(self.manager.config_loc(name), "data")
            shutil.copytree(data, data_src)
        self.manager.edit_config(name, name=name)

    def edit_simulation(self, name, **opts):
        self.check_simulation(name)
        self.manager.edit_config(name, **opts)

    def simulation_config(self, name):
        self.check_simulation(name)
        return self.manager.load_config(name)

    def simulation_output_folder(self, name):
        loc = self.manager.config_loc(name)
        return os.path.join(loc, "output")

    def simulation_ran(self, name):
        self.check_simulation(name)
        return os.path.isdir(self.simulation_output_folder(name))

    def load_simulation(self, name):

        self.check_simulation(name)
        curdir = os.getcwd()
        try:
            os.chdir(self.manager.config_loc(name))
            conf = self.manager.load_config(name)
            params = SimulationParameters(**conf.opt_dict)
            params.output_folder = self.simulation_output_folder(name)
            sim = Simulation(params)
            mpi_manager = sim.mpi_manager
            main = (mpi_manager is None) or (mpi_manager.world_rank == 0)
            if main and self.simulation_ran(name):
                sim.reload()
        finally:
            os.chdir(curdir)

        return sim

    def restart_simulation(self, name):
        shutil.rmtree(self.simulation_output_folder(name))
        self.run_simulation(name)

    def run_simulation(self, name):
        import sys

        sim = self.load_simulation(name)
        curdir = os.getcwd()
        try:
            os.chdir(self.manager.config_loc(name))

            log = sim.logger.log_file
            if sim.mpi_manager is not None:
                sim.run()
            elif isinstance(log, str):
                if sim.mpi_manager is not None:
                    if not os.path.isdir(os.path.dirname(log)):
                        os.makedirs(os.path.dirname(log))
                    sim.mpi_manager.wait()
                else:
                    if not os.path.isdir(os.path.dirname(log)):
                        os.makedirs(os.path.dirname(log))
                try:
                    with open(log, "w+", buffering=1) as log_stream:
                        sim.logger.log_file = log_stream
                        sout = sys.stdout
                        serr = sys.stderr
                        sys.stdout = log_stream
                        sys.stderr = log_stream
                        sim.run()
                finally:
                    sim.logger.log_file = log
                    sys.stdout = sout
                    sys.stderr = serr
            else:
                sim.run()

        finally:
            os.chdir(curdir)

    def export_simulation(self, name, path):
        self.check_simulation(name)
        shutil.copytree(self.manager.config_loc(name), path)

    def list_archive(self):
        return self.archiver.list_configs()
    def archive_simulation(self, name):
        import datetime as dt

        self.check_simulation(name)
        old_loc = self.manager.config_loc(name)
        new_name = dt.datetime.now().isoformat().split(".")[0].replace("/", "_").replace(":", "_")
        new_loc = self.archiver.config_loc(name+"_"+new_name)
        os.rename(old_loc, new_loc)
    def archive_config(self, name):
        return self.archiver.load_config(name)
    def archive_output_folder(self, name):
        loc = self.archiver.config_loc(name)
        return os.path.join(loc, "output")
    def check_archive(self, name):
        if not self.archiver.check_config(name):
            raise IOError("No archived simulation {}".format(name))
    def archived_simulation_ran(self, name):
        self.check_archive(name)
        return os.path.isdir(self.archive_output_folder(name))
    def load_archive(self, name):
        self.check_archive(name)
        conf = self.archiver.load_config(name)
        params = SimulationParameters(**conf.opt_dict)
        params.output_folder = self.archive_output_folder(name)
        sim = Simulation(params)
        if self.archived_simulation_ran(name):
            sim.reload()
        return sim