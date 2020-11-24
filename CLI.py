"""
Defines the command-line interface to RynLib
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from RynLib import VERSION_NUMBER
from RynLib.Interface import *

class CLI:

    command_groups = ("config", "sim", "pot")
    command_prefix = 'cli_method_'
    def __init__(self, group=None, command=None):
        if group is None or command is None:
            self.argv = sys.argv
            parser = argparse.ArgumentParser()
            parser.add_argument("group", type=str)
            parser.add_argument("command", type=str, default='', nargs="?")
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
        handle_unkown_opts = False
        unkown_opt_key = None
        for arg in spec:
            if len(arg) > 1:
                arg_name, arg_dict = arg
            else:
                arg_name = arg[0]
                arg_dict = {}
            if arg_dict == "OPTDICT":
                handle_unkown_opts = True
                unkown_opt_key = arg_name
            elif 'dest' in arg_dict:
                keys.append(arg_dict['dest'])
            else:
                keys.append(arg_name)
            if arg_dict != "OPTDICT":
                parser.add_argument(arg_name, **arg_dict)

        if not handle_unkown_opts:
            args = parser.parse_args()
        else:
            args, unknown = parser.parse_known_args()
            opt_dict = {}
            for k in unknown:
                if not k.startswith("--") or "=" not in k:
                    raise ValueError("Option '{}' is expected to look like '--<key>=<value>'".format(k))
                key, val = k.split('=', 1)
                key = key.strip("--")
                try:
                    val = int(val)
                except TypeError:
                    try:
                        val = float(val)
                    except TypeError:
                        pass
                opt_dict[key] = val

        opts = {k: getattr(args, k) for k in keys}
        if unkown_opt_key is not None:
            opts[unkown_opt_key] = opt_dict
        return {k:o for k,o in opts.items() if not (isinstance(o, str) and o=="")}

    def get_command(self, group=None, cmd=None):
        if group is None:
            group = self.group
        if cmd is None:
            cmd = self.cmd
        try:
            fun = getattr(self, self.command_prefix + group + "_" + cmd.replace("-", "_"))
        except AttributeError:
            fun = "Unknown command '{}' for command group '{}'".format(cmd.replace("_", "-"), group)
        return fun

    def get_help(self):
        from collections import OrderedDict

        blocks = []
        if self.group == "":
            return None
        elif self.group == "all":
            blocks.append(self.help_doc.replace(" ", "  "))
            groups = OrderedDict((k, OrderedDict()) for k in self.command_groups)
        else:
            groups = OrderedDict([(self.group, OrderedDict())])

        indent="    "
        template = "{group}:\n{commands}"
        bleps = self.command_prefix.count("_")
        if self.cmd == "":
            for k in vars(type(self)):
                for g in groups:
                    if k.startswith(self.command_prefix+g):
                        groups[g][k.split("_", bleps+1)[-1].replace("_", "-")] = getattr(self, k)
        else:
            template = "{group}{commands}"
            indent = "  "
            groups[self.group][self.cmd] = self.get_command()

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

    #region Config Methods
    def cli_method_config_build_libs(self):
        """
        Builds (or rebuilds) the libraries that the container uses. Shouldn't need to be called outside of Dockerfile/Singularity.def
        """
        get_bool = lambda s: False if s == 'False' else bool(s)
        parse_dict = self.get_parse_dict(
            ("--rebuild", dict(default=False, type=get_bool, dest='rebuild')),
        )
        RynLib.build_libs(**parse_dict)

    def cli_method_config_run_tests(self):
        """
        Runs the unit tests distributed with the package. Can be run in debug mode or validation mode.
        """
        get_bool = lambda s: False if s == 'False' else bool(s)
        parse_dict = self.get_parse_dict(
            ("--debug", dict(default=False, type=get_bool, dest='debug')),
            ("--testdir", dict(default="", type=str, dest='test_dir')),
            ("--name", dict(default="", type=str, dest='name')),
            ("--suite", dict(default="", type=str, dest='suite'))
        )
        RynLib.run_tests(**parse_dict)

    def cli_method_config_set_config(self):
        """
        Set configuration options for RynLib -- currently inactive
        """
        parse_dict = self.get_parse_dict(
            ("--simdir", dict(default="", type=str, dest='potential_directory')),
            ("--potdir", dict(default="", type=str, dest='simulation_directory')),
            ("--env", dict(default="singularity", type=str, dest='containerizer'))
        )
        RynLib.edit_config(**parse_dict)

    def cli_method_config_reset(self):
        '''
        Resets RynLib configuration to its default state. Useful for updates.
        '''
        RynLib.reset_config()

    def cli_method_config_install_mpi(self):
        '''
        Installs MPI. Shouldn't need to be called outside of Dockerfile/Singularity.def
        '''
        parse_dict = self.get_parse_dict(
            ("--version", dict(default="", type=str, dest='mpi_version')),
            ("--imp", dict(default="", type=str, dest='mpi_implementation'))
        )
        RynLib.install_MPI(**parse_dict)

    def cli_method_config_reload_dumpi(self):
        """
        Rebuilds just the Dumpi library that handles the MPI communication
        """
        RynLib.reload_dumpi()

    def cli_method_config_configure_mpi(self):
        """
        Installs MPI and rebuilds Dumpi
        """
        RynLib.configure_mpi()

    #endregion

    #region Simulation Methods
    def cli_method_sim_list(self):
        """Lists the simulations that have been added"""
        SimulationInterface.list_simulations()

    def cli_method_sim_add(self):
        """Adds a new simulation. Args: NAME SRC"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("src",)
            # ("--config", dict(default="", type=str, dest='config_file'))
        )
        SimulationInterface.add_simulation(**parse_dict)

    def cli_method_sim_remove(self):
        """Removes a simulation. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.remove_simulation(**parse_dict)

    def cli_method_sim_copy(self):
        """Copies a simulation. Args: NAME NEW_NAME"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("new_name",)
        )
        SimulationInterface.copy_simulation(**parse_dict)

    def cli_method_sim_edit(self):
        """Copies a simulation. Args: NAME [--OPTS=VALS]"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--optfile", dict(default="", type=str, dest='optfile')),
            ("opts", "OPTDICT")
        )
        SimulationInterface.edit_simulation(**parse_dict)

    def cli_method_sim_run(self):
        """Runs a simulation. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.run_simulation(**parse_dict)

    def cli_method_sim_restart(self):
        """Restarts a stopped simulation. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.restart_simulation(**parse_dict)

    def cli_method_sim_status(self):
        """Gets the status of a simulation. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.simulation_status(**parse_dict)

    def cli_method_sim_list_samplers(self):
        """Lists the added importance samplers"""
        SimulationInterface.list_samplers()

    def cli_method_sim_list_archive(self):
        """Lists the archived simulations"""
        SimulationInterface.list_archive()

    def cli_method_sim_archive(self):
        """Archives a simulation. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.archive_simulation(**parse_dict)

    def cli_method_sim_add_sampler(self):
        """Adds an importance sampler. Args: NAME SRC --config=CONFIG_FILE"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("source",)
            # ("--config", dict(default="", type=str, dest='config_file'))
        )
        SimulationInterface.add_sampler(**parse_dict)

    def cli_method_sim_remove_sampler(self):
        """Removes an importance sampler. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        SimulationInterface.remove_sampler(**parse_dict)

    def cli_method_sim_test_sampler(self):
        """Tests an importance sampler. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--input", dict(default="", type=str, dest='input_file'))
        )
        SimulationInterface.test_sampler(**parse_dict)
    def cli_method_sim_test_sampler_mpi(self):
        """Tests an importance sampler. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--input", dict(default="", type=str, dest='input_file'))
        )
        SimulationInterface.test_sampler_mpi(**parse_dict)

    def cli_method_sim_test_HO(self):
        """Runs a harmonic oscillator DMC as a test"""
        SimulationInterface.test_HO()
    #endregion

    #region Potential Methods
    def cli_method_pot_list(self):
        """Lists the potentials that have been added"""
        PotentialInterface.list_potentials()

    def cli_method_pot_add(self):
        """Adds a new potential. Args: NAME SRC"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("src",)
            # ("--config", dict(default="", type=str, dest='config_file')),
            # ("--data", dict(default="", type=str, dest='data')),
            # ("--test", dict(default="", type=str, dest='test_file'))
        )
        PotentialInterface.add_potential(**parse_dict)

    def cli_method_pot_remove(self):
        """Removes a potential. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        PotentialInterface.remove_potential(**parse_dict)

    def cli_method_pot_import(self):
        """Imports a potential from an existing archive. Args: NAME SRC"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("src",)
        )
        PotentialInterface.import_potential(**parse_dict)
    def cli_method_pot_export(self):
        """Export a potential to an archive. Args: NAME SRC"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("dest",)
        )
        PotentialInterface.export_potential(**parse_dict)

    def cli_method_pot_compile(self):
        """Ensures that a potential has been compiled. Args: NAME"""
        get_bool = lambda s: False if s == 'False' else bool(s)
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--recompile", dict(default=False, type=get_bool, dest='recompile'))
        )
        PotentialInterface.compile_potential(**parse_dict)

    def cli_method_pot_status(self):
        """Checks the status of a potential. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",)
        )
        PotentialInterface.potential_status(**parse_dict)

    def cli_method_pot_configure_entos(self):
        """Configures the built in Entos potential"""
        PotentialInterface.configure_entos()

    def cli_method_pot_test_entos(self):
        """Tests the built in Entos potential"""
        RynLib.test_entos()

    def cli_method_pot_test_HO(self):
        """Tests the built in Harmonic Oscillator potential"""
        RynLib.test_HO()

    def cli_method_pot_test_entos_mpi(self):
        """Tests Entos via MPI"""
        parse_dict = self.get_parse_dict(
            ("--per_core", dict(default=5, type=int, dest="walkers_per_core")),
            ("--disp", dict(default=.5, type=int, dest="displacement_radius")),
            ("--its", dict(default=5, type=int, dest="iterations")),
            ("--steps", dict(default=5, type=int, dest="steps_per_call"))
        )
        RynLib.test_entos_mpi(**parse_dict)

    def cli_method_pot_test_ho_mpi(self):
        """Tests the HO via MPI"""
        parse_dict = self.get_parse_dict(
            ("--per_core", dict(default=5, type=int, dest="walkers_per_core")),
            ("--disp", dict(default=.5, type=int, dest="displacement_radius")),
            ("--its", dict(default=5, type=int, dest="iterations")),
            ("--steps", dict(default=5, type=int, dest="steps_per_call"))
        )
        RynLib.test_ho_mpi(**parse_dict)

    def cli_method_pot_test(self):
        """Tests a generic potential. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--input", dict(default="", type=str, dest='input_file'))
        )
        PotentialInterface.test_potential(**parse_dict)

    def cli_method_pot_test_mpi(self):
        """Tests a generic potential under MPI. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--input", dict(default="", type=str, dest='input_file'))
        )
        PotentialInterface.test_potential_mpi(**parse_dict)

    def cli_method_pot_test_serial(self):
        """Tests a generic potential under looping. Args: NAME"""
        parse_dict = self.get_parse_dict(
            ("name",),
            ("--input", dict(default="", type=str, dest='input_file'))
        )
        PotentialInterface.test_potential_serial(**parse_dict)
    #endregion

    def run(self):
        res = self.get_command()
        if not isinstance(res, str):
            res = res()
        else:
            print(res)
        return res

    def help(self, print_help=True):
        try:
            sys.argv.pop(1)
        except IndexError:
            pass
        res = self.get_help()
        if print_help and res is not None:
            print(res)
        return res

    help_doc = """
    rynlib [--<flags>] GRP CMD [ARGS] runs RynLib with the specified command
    Flags:
     --help: print this help message
     --help <grp>:
      all: list all available commands
      grp: list commands in grp
     --output=<FILE>: bind stdout to FILE
     --error=<FILE>: bind stderr to FILE
     --script=<FILE>: run FILE in the RynLib environment
     --root=<PATH>: specify the root directory to do resource resolution from
     --interact: start an interactive session after running the command
     --fulltb: turn on full tracebacks
     --noomp: turn off OpenMP parallelism
     --thomp: specify the number of OpenMP threads that were set outside the program (if != os.cpu_count)
     --notbb: turn off Threaded Building Blocks parallelism
     --thtbb: specify the number of TBB threads that were set outside the program (if != os.cpu_count)
    """.replace("    ", "").strip()
    @classmethod
    def run_command(cls, parse):
        # detect whether interactive run or not
        interact = parse.interact or (len(sys.argv) == 1 and not parse.help and not parse.script)

        # handle runtime flags
        if parse.thomp>0:
            RynLib.flags['OpenMPThreads'] = parse.thomp
        if parse.thtbb>0:
            RynLib.flags['TBBThreads'] = parse.thtbb
        if parse.noomp:
            RynLib.flags['OpenMPThreads'] = False
        if parse.notbb:
            RynLib.flags['TBBThreads'] = False
        if parse.pyp:
            RynLib.flags['multiprocessing'] = True

        # set root directory
        root = parse.root
        if isinstance(root, str) and len(root) > 0:
            RynLib.root = root

        # in interactive/script envs we expose stuff
        if parse.script or interact:
             import RynLib.DoMyCode as DMC, RynLib.PlzNumbers as Potentials, RynLib.Dumpi as MPI

             sys.path.insert(0, os.getcwd())
             interactive_env = {
                 "__name__": "RynLib.script",
                 'DMC': DMC, 'SimulationManager': DMC.SimulationManager,
                 'Potentials': Potentials, 'PotentialManager': Potentials.PotentialManager,
                 'MPI' : MPI, 'MPIManager': MPI.MPIManager
                }
        # in a script environment we just read in the script and run it
        if parse.script:
            with open(parse.script) as script:
                src = script.read()
                src = compile(src, parse.script, 'exec')
            interactive_env["__file__"] = parse.script
            exec(src, interactive_env, interactive_env)
        elif parse.help:
            if len(sys.argv) == 1:
                print(cls.help_doc.splitlines()[0])
            group = sys.argv[1] if len(sys.argv) > 1 else ""
            command = sys.argv[2] if len(sys.argv) > 2 else ""
            CLI(group=group, command=command).help()
        elif len(sys.argv) > 1:
            CLI().run()
        if interact:
            import code
            code.interact(
                banner="RynLib Interactive Session (version {})".format(VERSION_NUMBER),
                readfunc=None,
                local=interactive_env,
                exitmsg=None
            )

    @classmethod
    def run_parse(cls, parse, unknown):
        if not parse.ignore:
            # if parse.update or parse.rebuild:
            #     import subprocess
            #
            #     CLI.update_lib(rebuild=parse.rebuild)
            #
            #     sys.argv = [sys.argv[0]]
            #     if parse.output != "":
            #         sys.argv.append("--output="+parse.output)
            #     if parse.error != "":
            #         sys.argv.append("--error="+parse.error)
            #     if parse.interact:
            #         sys.argv.append("--interact")
            #     if parse.interact:
            #         sys.argv.append("--help")
            #     if parse.no_openmp:
            #         sys.argv.append("--nomp")
            #     sys.argv += unknown
            #     subprocess.call([sys.executable, *sys.argv])
            # else:
            stdout = sys.stdout
            stderr = sys.stderr
            sys.argv = [sys.argv[0]] + unknown
            # print(sys.argv)
            try:
                if parse.output != "":
                    with open(parse.output, "w+", buffering=1) as out:
                        sys.stdout = out
                        if parse.error != "":
                            with open(parse.error, "w+", buffering=1) as err:
                                sys.stderr = err
                                cls.run_command(parse)
                        else:
                            sys.stderr = out
                            cls.run_command(parse)
                else:
                    cls.run_command(parse)
            finally:
                sys.stdout = stdout
                sys.stderr = stderr

    @classmethod
    def parse_and_run(cls):
        if "-h" in sys.argv:
            print("use rynlib --help to get help")
            return

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--update", default=False, action='store_const', const=True, dest="update")
        parser.add_argument("--rebuild", default=False, action='store_const', const=True, dest="rebuild")
        parser.add_argument("--pyp", default=False, action='store_const', const=True, dest="pyp",
                            help='turn on multiprocessing.Pool parallelism'
                            )
        parser.add_argument("--noomp", default=False, action='store_const', const=True, dest="noomp",
                            help='turn off OpenMP parallelism'
                            )
        parser.add_argument("--thomp", default=0, type=int, dest="thomp",
                            help='number of OpenMP threads to use'
                            )
        parser.add_argument("--notbb", default=False, action='store_const', const=True, dest="notbb",
                            help='turn off TBB parallelism'
                            )
        parser.add_argument("--thtbb", default=0, type=int, dest="thtbb",
                            help='number of OpenMP threads to use'
                            )
        parser.add_argument("--output", default="", type=str, dest="output",
                            help='stdout file to write to'
                            )
        parser.add_argument("--error", default="", type=str, dest="error",
                            help='stderr file to write to'
                            )
        parser.add_argument("--script", default="", type=str, dest="script",
                            help='a script to run'
                            )
        parser.add_argument("--root", default="", type=str, dest="root",
                            help='the path relative to use as the root directory for config lookup'
                            )
        parser.add_argument("--interact", default=False, action='store_const', const=True, dest="interact",
                            help='start an interactive session after running'
                            )
        parser.add_argument("--help", default=False, action='store_const', const=True, dest="help")
        parser.add_argument("--pass", default=False, action='store_const', const=True, dest="ignore")
        parser.add_argument("--fulltb", default=False, action='store_const', const=True, dest="full_traceback")
        new_argv = []
        for k in sys.argv[1:]:
            if not k.startswith("--"):
                break
            new_argv.append(k)
        unknown = sys.argv[1+len(new_argv):]
        sys.argv = [sys.argv[0]]+new_argv
        parse = parser.parse_args()

        if parse.full_traceback:
            cls.run_parse(parse, unknown)
        else:
            error = None
            try:
                cls.run_parse(parse, unknown)
            except Exception as e:
                error = e
            if error is not None:
                print(error)

if __name__ == "__main__":
    CLI.parse_and_run()