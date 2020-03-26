"""
Defines the command-line interface to RynLib
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from RynLib.Interface import *

class CLI:
    def __init__(self):
        self.argv = sys.argv
        parser = argparse.ArgumentParser()
        parser.add_argument("group", type=str)
        parser.add_argument("command", type=str)
        parse, unknown = parser.parse_known_args()
        self.group = parse.group
        self.cmd = parse.command
        sys.argv = [sys.argv[0]] + unknown

    def get_parse_dict(self, *spec):
        parser = argparse.ArgumentParser()
        keys = []
        for arg in spec:
            if 'dest' in arg[1]:
                keys.append(arg[1]['dest'])
            else:
                keys.append(arg[0])
            parser.add_argument(arg[0], **arg[1])
        args = parser.parse_args()
        return {k: getattr(args, k) for k in keys}

    def config_set_config(self):
        """
        Set configuation options for RynLib
        :return:
        :rtype:
        """
        parse_dict = self.get_parse_dict(
            ("--simdir", dict(default="", type=str, dest='potential_directory')),
            ("--potdir", dict(default="", type=str, dest='simulation_directory')),
            ("--env", dict(default="singularity", type=str, dest='containerizer'))
        )
        GeneralConfig.edit_config(**parse_dict)

    def config_update_lib(self):
        GeneralConfig.update_lib()

    def sim_add(self):
        """
        Add a simulation to RynLib
        :return:
        :rtype:
        """
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--config", dict(default="", type=str, dest='config'))
        )
        SimulationInterface.add_simulation(**parse_dict)

    def sim_set_config(self):
        """
        Add a simulation to RynLib
        :return:
        :rtype:
        """
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--config", dict(default="", type=str, dest='config'))
        )
        SimulationInterface.set_config(**parse_dict)

    def run(self):
        getattr(self, self.group + "_" + self.cmd)()


if __name__ == "__main__":
    if sys.argv[1] == "-i":
        import code
        code.interact(banner=None, readfunc=None, local=None, exitmsg=None)
    else:
        CLI().run()