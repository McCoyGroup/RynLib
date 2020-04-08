from ..RynUtils import ConfigManager
from .Simulation import Simulation, SimulationParameters
import os, shutil

__all__ = [
    "SimulationManager"
]

class SimulationManager:
    def __init__(self, config_dir=None):
        if config_dir is None:
            from ..Interface import RynLib
            config_dir = RynLib.get_conf().simulation_directory
        self.manager = ConfigManager(config_dir)

    def list_simulations(self):
        return self.manager.list_configs()

    def remove_simulation(self, name):
        self.manager.remove_config(name)

    def add_simulation(self, name, config_file = None, data=None, **opts):
        self.manager.add_config(name, config_file = config_file, **opts)
        if data is not None:
            data_src = os.path.join(self.manager.config_loc(name), os.path.basename(data))
            shutil.copytree(data, data_src)
        self.manager.edit_config(name, name=name)

    def edit_simulation(self, name, **opts):
        self.manager.edit_config(name, **opts)

    def simulation_config(self, name):
        return self.manager.load_config(name)

    def simulation_output_folder(self, name):
        loc = self.manager.config_loc(name)
        return os.path.join(loc, "data")

    def simulation_ran(self, name):
        return os.path.isdir(self.simulation_output_folder(name))

    def load_simulation(self, name):
        if self.simulation_ran(name):
            sim = Simulation.reload(output_folder=self.simulation_output_folder(name))
        else:
            conf = self.manager.load_config(name)
            params = SimulationParameters(**conf.opt_dict)
            params.output_folder = self.simulation_output_folder(name)
            sim = Simulation(params)

        return sim

    def restart_simulation(self, name):
        shutil.rmtree(self.simulation_output_folder(name))
        self.run_simulation(name)

    def run_simulation(self, name):
        import sys

        sim = self.load_simulation(name)

        log = sim.logger.log_file
        if isinstance(log, str):
            if not os.path.isdir(os.path.dirname(log)):
                os.makedirs(os.path.dirname(log))
            try:
                with open(log, "w+") as log_stream:
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

    def simulation_data(self, name, key):
        """Loads a simulation and returns its data...I guess?

        :param name:
        :type name:
        :param key:
        :type key:
        :return:
        :rtype:
        """

        raise NotImplemented

    def export_simulation(self, name, path):
        shutil.copytree(self.manager.config_loc(name), path)