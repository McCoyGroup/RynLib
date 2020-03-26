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
            if len(arg) > 1:
                arg_name, arg_dict = arg
            else:
                arg_name = arg
                arg_dict = {}
            if 'dest' in arg_dict:
                keys.append(arg_dict['dest'])
            else:
                keys.append(arg_name)
            parser.add_argument(arg_name, **arg_dict)
        args = parser.parse_args()
        return {k: getattr(args, k) for k in keys}

    def config_run_tests(self):
        GeneralConfig.run_tests()

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

    def config_install_mpi(self):
        GeneralConfig.install_MPI()

    def config_update_lib(self):
        GeneralConfig.update_lib()

    def config_update_testing_framework(self):
        GeneralConfig.update_testing_framework()

    def sim_list(self):
        SimulationInterface.list_simulations()

    def sim_add(self):
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--config", dict(default="", type=str, dest='config'))
        )
        SimulationInterface.add_simulation(**parse_dict)

    def sim_remove(self):
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.remove_simulation(**parse_dict)

    def sim_run(self):
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.run_simulation(**parse_dict)

    def pot_list(self):
        PotentialInterface.list_potentials()

    def pot_add(self):
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--source", dict(default="", type=str, dest='src')),
            ("--config", dict(default="", type=str, dest='config'))
        )
        PotentialInterface.add_potential(**parse_dict)

    def pot_remove(self):
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        PotentialInterface.remove_potential(**parse_dict)

    def pot_compile(self):
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        PotentialInterface.compile_potential(**parse_dict)

    def run(self):
        getattr(self, self.group + "_" + self.cmd)()


if __name__ == "__main__":
    print(sys.argv[1], sys.argv[2])
    if sys.argv[1] == "interact":
        import code
        code.interact(banner=None, readfunc=None, local=None, exitmsg=None)
    else:
        CLI().run()