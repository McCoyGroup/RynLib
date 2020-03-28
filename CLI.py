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
                arg_name = arg[0]
                arg_dict = {}
            if 'dest' in arg_dict:
                keys.append(arg_dict['dest'])
            else:
                keys.append(arg_name)
            parser.add_argument(arg_name, **arg_dict)
        args = parser.parse_args()
        opts = {k: getattr(args, k) for k in keys}
        return {k:o for k,o in opts.items() if not (isinstance(o, str) and o=="")}

    def config_build_libs(self):
        RynLib.build_libs()

    def config_run_tests(self):
        RynLib.run_tests()

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
        RynLib.edit_config(**parse_dict)

    def config_reset(self):
        RynLib.reset_config()

    def config_install_mpi(self):
        RynLib.install_MPI()

    def config_reload_dumpi(self):
        RynLib.reload_dumpi()

    def config_configure_mpi(self):
        RynLib.configure_mpi()

    def config_test_mpi(self):
        RynLib.test_mpi()

    def config_update_lib(self):
        RynLib.update_lib()

    def config_update_testing_framework(self):
        RynLib.update_testing_framework()

    def config_test_entos(self):
        RynLib.test_entos()

    def config_test_HO(self):
        RynLib.test_HO()

    def config_test_entos_mpi(self):
        parse_dict = self.get_parse_dict(
            ("--per_core", dict(default=5, type=int, dest="walkers_per_core")),
            ("--disp", dict(default=.5, type=int, dest="displacement_radius")),
            ("--its", dict(default=5, type=int, dest="iterations")),
            ("--steps", dict(default=5, type=int, dest="steps_per_call"))
        )
        RynLib.test_entos_mpi(**parse_dict)

    def config_test_ho_mpi(self):
        parse_dict = self.get_parse_dict(
            ("--per_core", dict(default=5, type=int, dest="walkers_per_core")),
            ("--disp", dict(default=.5, type=int, dest="displacement_radius")),
            ("--its", dict(default=5, type=int, dest="iterations")),
            ("--steps", dict(default=5, type=int, dest="steps_per_call"))
        )
        RynLib.test_ho_mpi(**parse_dict)

    def sim_list(self):
        SimulationInterface.list_simulations()

    def sim_add(self):
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--config", dict(default="", type=str, dest='config_file'))
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

    def sim_status(self):
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.simulation_status(**parse_dict)

    def sim_list_samplers(self):
        SimulationInterface.list_samplers()

    def sim_add_sampler(self):
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--config", dict(default="", type=str, dest='config_file')),
            ("--src", dict(default="", type=str, dest='source'))
        )
        SimulationInterface.add_sampler(**parse_dict)

    def sim_remove_sampler(self):
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.remove_sampler(**parse_dict)

    def pot_list(self):
        PotentialInterface.list_potentials()

    def pot_add(self):
        parse_dict = self.get_parse_dict(
            ("name",),
            ("src",),
            ("--config", dict(default="", type=str, dest='config_file')),
            ("--data", dict(default="", type=str, dest='data')),
            ("--test", dict(default="", type=str, dest='test_file'))
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

    def pot_status(self):
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        PotentialInterface.potential_status(**parse_dict)

    def pot_test(self):
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--in", dict(default="", type=str, dest='input_file'))
        )
        PotentialInterface.test_potential(**parse_dict)

    def pot_test_mpi(self):
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--in", dict(default="", type=str, dest='input_file'))
        )
        PotentialInterface.test_potential_mpi(**parse_dict)

    def pot_configure_entos(self):
        PotentialInterface.configure_entos()

    def run(self):
        getattr(self, self.group + "_" + self.cmd.replace("-", "_"))()

if __name__ == "__main__":
    if sys.argv[1] == "interact":
        import code
        code.interact(banner=None, readfunc=None, local=None, exitmsg=None)
    else:
        CLI().run()