from ..RynUtils import ConfigManager
from .Potential import Potential
import os, shutil

__all__ = [
    "PotentialManager"
]

class PotentialManager:
    def __init__(self, config_dir=None):
        if config_dir is None:
            from ..Interface import GeneralConfig
            config_dir = GeneralConfig.get_conf().potential_directory
        self.manager = ConfigManager(config_dir)

    def list_potentials(self):
        return self.manager.list_configs()

    def remove_potential(self, name):
        self.manager.remove_config(name)

    def add_potential(self, name, src, config_file = None, **opts):
        self.manager.add_config(name, config_file = config_file, **opts)
        new_src = os.path.join(self.manager.config_loc(name), os.path.basename(src))
        if os.path.isdir(src):
            shutil.copy(src, new_src)
        else:
            shutil.copyfile(src, new_src)
        self.manager.edit_config(name, potential_source=new_src)

    def potential_config(self, name):
        return self.manager.load_config(name)

    def load_potential(self, name):
        conf = self.manager.load_config(name)
        params = conf.opt_dict
        out_dir = self.manager.config_loc(name)
        params['out_dir'] = out_dir
        return Potential(**params)

    def compile_potential(self, name):
        pot = self.load_potential(name)
        pot.caller # causes the potential to compile what needs to be compiled