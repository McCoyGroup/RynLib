from ..RynUtils import ConfigManager
from .Simulation import Simulation, SimulationParameters
import os, shutil

__all__ = [
    "SimulationManager"
]

class SimulationManager:
    def __init__(self, config_dir = os.path.expanduser("~/Desktop/simulations")):
        self.manager = ConfigManager(config_dir)

    def list_simulations(self):
        return self.manager.list_configs()

    def remove_simulation(self, name):
        self.manager.remove_config(name)

    def add_simulation(self, name, config_file = None, **opts):
        self.manager.add_config(name, config_file = config_file, **opts)

    def simulation_output_folder(self, name):
        loc = self.manager.config_loc(name)
        return os.path.join(loc, "dmc_data")

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
        sim = self.load_simulation(name)

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