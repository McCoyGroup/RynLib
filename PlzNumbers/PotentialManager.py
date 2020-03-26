from ..RynUtils import ConfigManager
from .Potential import Potential
import os, shutil

__all__ = [
    "PotentialManager"
]

class PotentialManager:
    def __init__(self, config_dir=None):
        if config_dir is None:
            from ..Interface import RynLib
            config_dir = RynLib.get_conf().potential_directory
        self.manager = ConfigManager(config_dir)

    def list_potentials(self):
        return self.manager.list_configs()

    def remove_potential(self, name):
        self.manager.remove_config(name)

    def add_potential(self, name, src, config_file = None, static_source = False, **opts):
        self.manager.add_config(name, config_file = config_file, **opts)
        if not static_source:
            new_src = os.path.join(self.manager.config_loc(name), "raw_source", os.path.basename(src))
            os.makedirs(os.path.join(self.manager.config_loc(name), "raw_source"))
            if os.path.isdir(src):
                shutil.copytree(src, new_src)
            else:
                shutil.copyfile(src, new_src)
            self.manager.edit_config(name, name=name, potential_source=new_src, static_source=False)
        else:
            self.manager.edit_config(name, name=name, potential_source=src, static_source=True)

    def potential_config(self, name):
        return self.manager.load_config(name)

    def load_potential(self, name):
        if name == "entos" and "entos" not in self.list_potentials():
            from ..Interface import PotentialInterface
            PotentialInterface.configure_entos()
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