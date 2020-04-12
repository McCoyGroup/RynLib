"""
Defines the command-line interface to RynLib
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from RynLib.Interface import *

class CLI:
    def __init__(self, group=None, command=None):
        if group is None or command is None:
            self.argv = sys.argv
            parser = argparse.ArgumentParser()
            parser.add_argument("group", type=str)
            parser.add_argument("command", type=str)
            parse, unknown = parser.parse_known_args()
            self.group = parse.group
            self.cmd = parse.command
            sys.argv = [sys.argv[0]] + unknown
        else:
            self.group = group
            self.cmd = command

    @classmethod
    def update_lib(self, rebuild=False):
        """
        We can use this to dynamically pull changes off of GitHub right before running the code
        """
        RynLib.update_lib()
        RynLib.build_libs(rebuild=rebuild)

    def get_parse_dict(self, *spec):
        sys.argv[0] = self.group + " " + self.cmd
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

    def get_command(self, group=None, cmd=None):
        if group is None:
            group = self.group
        if cmd is None:
            cmd = self.cmd
        try:
            fun = getattr(self, group + "_" + cmd.replace("-", "_"))
        except AttributeError:
            fun = "Unknown command '{}' for command group '{}'".format(cmd.replace("_", "-"), group)
        return fun

    def get_help(self):
        from collections import OrderedDict

        if self.group == "":
            groups = OrderedDict((k, OrderedDict()) for k in ("config", "sim", "pot"))
        else:
            groups = OrderedDict((self.group, OrderedDict()))

        indent="    "
        template = "{group}:\n{commands}"
        if self.cmd == "":
            for k in vars(type(self)):
                for g in groups:
                    if k.startswith(g):
                        groups[g][k.split("_", 1)[1].replace("_", "-")] = getattr(self, k)
        else:
            template = "{group}{commands}"
            indent = "  "
            groups[self.group][self.cmd] = self.get_command()

        blocks = []
        make_command_info = lambda name, fun, indent: "{0}{1}{3}{0}  {2}".format(
            indent,
            name,
            "" if fun.__doc__ is None else fun.__doc__.strip(),
            "\n" if fun.__doc__ is not None else ""
            )
        for g in groups:
            blocks.append(
                template.format(
                    group = g,
                    commands = "\n".join(make_command_info(k, f, indent) for k, f in groups[g].items())
                )
            )
        return "\n\n".join(blocks)

    def config_build_libs(self):
        """
        Builds (or rebuilds) the libraries that the container uses. Shouldn't need to be called outside of Dockerfile/Singularity.def
        """
        get_bool = lambda s: False if s == 'False' else bool(s)
        parse_dict = self.get_parse_dict(
            ("--rebuild", dict(default=False, type=get_bool, dest='rebuild')),
        )
        RynLib.build_libs(**parse_dict)

    def config_run_tests(self):
        """
        Currently inactive; should hook into the unit tests that we've bundled, once we have enough test data
        """
        RynLib.run_tests()

    def config_set_config(self):
        """
        Set configuration options for RynLib -- currently inactive
        """
        parse_dict = self.get_parse_dict(
            ("--simdir", dict(default="", type=str, dest='potential_directory')),
            ("--potdir", dict(default="", type=str, dest='simulation_directory')),
            ("--env", dict(default="singularity", type=str, dest='containerizer'))
        )
        RynLib.edit_config(**parse_dict)

    def config_reset(self):
        '''
        Resets RynLib configuration to its default state. Useful for updates.
        '''
        RynLib.reset_config()

    def config_install_mpi(self):
        '''
        Installs MPI. Shouldn't need to be called outside of Dockerfile/Singularity.def
        '''
        parse_dict = self.get_parse_dict(
            ("--version", dict(default="", type=str, dest='mpi_version')),
            ("--imp", dict(default="", type=str, dest='mpi_implementation'))
        )
        RynLib.install_MPI(**parse_dict)

    def config_reload_dumpi(self):
        """
        Rebuilds just the Dumpi library that handles the MPI communication
        """
        RynLib.reload_dumpi()

    def config_configure_mpi(self):
        """
        Installs MPI and rebuilds Dumpi
        """
        RynLib.configure_mpi()

    def config_test_mpi(self):
        """
        Tests that MPI/Dumpi can initialize cleanly
        """
        RynLib.test_mpi()

    # def config_update_testing_framework(self):
    #     RynLib.update_testing_framework()

    def sim_list(self):
        """Lists the simulations that have been added"""
        SimulationInterface.list_simulations()

    def sim_add(self):
        """Adds a new simulation. Args: NAME SRC"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("src",)
            # ("--config", dict(default="", type=str, dest='config_file'))
        )
        SimulationInterface.add_simulation(**parse_dict)

    def sim_remove(self):
        """Removes a simulation. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.remove_simulation(**parse_dict)

    def sim_run(self):
        """Runs a simulation. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.run_simulation(**parse_dict)

    def sim_status(self):
        """Gets the status of a simulation. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.simulation_status(**parse_dict)

    def sim_list_samplers(self):
        """Lists the added importance samplers"""
        SimulationInterface.list_samplers()

    def sim_add_sampler(self):
        """Adds an importance sampler. Args: NAME SRC --config=CONFIG_FILE"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("source",)
            # ("--config", dict(default="", type=str, dest='config_file'))
        )
        SimulationInterface.add_sampler(**parse_dict)

    def sim_remove_sampler(self):
        """Removes an importance sampler. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.remove_sampler(**parse_dict)

    def sim_test_sampler(self):
        """Tests an importance sampler. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.test_sampler(**parse_dict)

    def sim_test_HO(self):
        """Runs a harmonic oscillator DMC as a test"""
        SimulationInterface.test_HO()

    def pot_list(self):
        """Lists the potentials that have been added"""
        PotentialInterface.list_potentials()

    def pot_add(self):
        """Adds a new potential. Args: NAME SRC"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("src",)
            # ("--config", dict(default="", type=str, dest='config_file')),
            # ("--data", dict(default="", type=str, dest='data')),
            # ("--test", dict(default="", type=str, dest='test_file'))
        )
        PotentialInterface.add_potential(**parse_dict)

    def pot_remove(self):
        """Removes a potential. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        PotentialInterface.remove_potential(**parse_dict)

    def pot_compile(self):
        """Ensures that a potential has been compiled. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        PotentialInterface.compile_potential(**parse_dict)

    def pot_status(self):
        """Checks the status of a potential. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        PotentialInterface.potential_status(**parse_dict)

    def pot_configure_entos(self):
        """Configures the built in Entos potential"""
        PotentialInterface.configure_entos()

    def pot_test_entos(self):
        """Tests the built in Entos potential"""
        RynLib.test_entos()

    def pot_test_HO(self):
        """Tests the built in Harmonic Oscillator potential"""
        RynLib.test_HO()

    def pot_test_entos_mpi(self):
        """Tests Entos via MPI"""
        parse_dict = self.get_parse_dict(
            ("--per_core", dict(default=5, type=int, dest="walkers_per_core")),
            ("--disp", dict(default=.5, type=int, dest="displacement_radius")),
            ("--its", dict(default=5, type=int, dest="iterations")),
            ("--steps", dict(default=5, type=int, dest="steps_per_call"))
        )
        RynLib.test_entos_mpi(**parse_dict)

    def pot_test_ho_mpi(self):
        """Tests the HO via MPI"""
        parse_dict = self.get_parse_dict(
            ("--per_core", dict(default=5, type=int, dest="walkers_per_core")),
            ("--disp", dict(default=.5, type=int, dest="displacement_radius")),
            ("--its", dict(default=5, type=int, dest="iterations")),
            ("--steps", dict(default=5, type=int, dest="steps_per_call"))
        )
        RynLib.test_ho_mpi(**parse_dict)

    def pot_test(self):
        """Tests a generic potential. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
            # ("--in", dict(default="", type=str, dest='input_file'))
        )
        PotentialInterface.test_potential(**parse_dict)

    def pot_test_mpi(self):
        """Tests a generic potential under MPI. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
            # ("--in", dict(default="", type=str, dest='input_file'))
        )
        PotentialInterface.test_potential_mpi(**parse_dict)

    def run(self):
        res = self.get_command()
        if not isinstance(res, str):
            res = res()
        return res

    def help(self, print_help=True):
        sys.argv.pop(1)
        res = self.get_help()
        if print_help:
            print(res)
        return res

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"--update", "--rebuild"}
        import subprocess
        sys.argv.pop(1)
        CLI.update_lib(rebuild=sys.argv[1]=="--rebuild")
        subprocess.call([sys.executable, *sys.argv])
    elif len(sys.argv) == 1 or sys.argv[1] == "interact":
        import code
        code.interact(banner="RynLib Interactive Session", readfunc=None, local=None, exitmsg=None)
    elif sys.argv[1] == "help":
        group = sys.argv[2] if len(sys.argv) > 2 else ""
        command = sys.argv[3] if len(sys.argv) > 3 else ""
        CLI(group=group, command=command).help()
    elif sys.argv[1] == "ignore":
        pass
    else:
        CLI().run()