import os, numpy as np, enum, sys
from ...RynUtils import Logger

__all__ = [
    "SimulationLogger"
]

class SimulationLogger:
    """
    A class for saving simulation data
    """

    class LogLevel(enum.Enum):
        BASIC = 1
        STATUS = 3
        STEPS = 5
        DATA = 6
        MPI = 7
        ALL = 10
        DEBUG = 100

    __props__ = [
        "output_folder",
        "log_file",
        "log_level",
        "checkpoints_folder",
        'wavefunctions_folder'
    ]
    def __init__(self,
                 simulation,
                 output_folder = None,
                 log_file = None,
                 log_level = 'DATA',
                 checkpoint_folder = "checkpoints",
                 wavefunctions_folder = 'wavefunctions'
                 ):
        """

        :param simulation:
        :type simulation: Simulation
        :param output_folder:
        :type output_folder: str
        :param log_file:
        :type log_file: str
        :param verbosity:
        :type verbosity: int | str
        """
        self.sim = simulation
        self.output_folder = output_folder

        if output_folder is None:
            output_folder = os.path.join(os.path.abspath("dmc_data"), self.sim.name)
        self.output_folder = output_folder
        if checkpoint_folder is not None:
            if not os.path.isdir(checkpoint_folder):
                checkpoint_folder = os.path.join(self.output_folder, checkpoint_folder)
            self.checkpoint_folder = checkpoint_folder
        else:
            self.checkpoint_folder = self.output_folder
        if wavefunctions_folder is not None:
            if not os.path.isdir(wavefunctions_folder):
                wavefunctions_folder = os.path.join(self.output_folder, wavefunctions_folder)
            self.wavefunctions_folder = wavefunctions_folder
        else:
            self.wavefunctions_folder = self.output_folder

        if log_file is None:
            log_file = os.path.join(self.output_folder, "log.txt")
        elif isinstance(log_file, str) and log_file.lower() == "stdout":
            log_file = sys.stdout
        elif isinstance(log_file, str) and log_file.lower() == "stderr":
            log_file = sys.stderr
        self.log_file = log_file

        verbosity = log_level
        if isinstance(verbosity, str):
            verbosity = verbosity.upper()
            verbosity = getattr(self.LogLevel, verbosity)
        self.verbosity = verbosity

        self.verb_int = self.verbosity.value if isinstance(self.verbosity, self.LogLevel) else verbosity
        self.logger = Logger(self.log_file, verbosity = self.verb_int)

    def log_print(self, *args, verbosity=None, **kwargs):
        if verbosity is None:
            verbosity = self.verb_int
        elif isinstance(verbosity, self.LogLevel):
            verbosity = verbosity.value
        self.logger.log_print(*args, verbosity=verbosity, **kwargs)

    # def snapshot(self, file="snapshot.pickle"):
    #     """Saves a snapshot of the simulation to file
    #
    #     :param file:
    #     :type file:
    #     :return:
    #     :rtype:
    #     """
    #     raise NotImplementedError("Turns out pickle doesn't like this")
    #
    #     # import pickle
    #     #
    #     # f = os.path.abspath(file)
    #     # if not os.path.isfile(f):
    #     #     if not os.path.isdir(self.output_folder):
    #     #         os.makedirs(self.output_folder)
    #     #     f = os.path.join(self.output_folder, file)
    #     # with open(f, "w+") as binary:
    #     #     pickle.dump(self, binary)

    def snapshot_params(self, file="checkpoint.json"):
        """Saves a snapshot of the params to a pickle

        :return:
        :rtype:
        """

        f = os.path.abspath(file)
        if not os.path.isfile(f):
            f = os.path.join(self.output_folder, file)
        out_dir = os.path.dirname(f)
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir)
            except FileExistsError:
                pass
        self.sim.params.serialize(self.sim, f)

    def snapshot_walkers(self, file="walkers{core}_{n}.npz", save_stepnum = True):
        """Saves a snapshot of the walkers to a pickle

        :return:
        :rtype:
        """

        n = "" if not save_stepnum else self.sim.counter.step_num
        core = self.sim.world_rank
        if core == 0:
            core = ""
        file = file.format(core=core, n=n)
        f = os.path.abspath(file)
        if file != f:
            f = os.path.join(self.checkpoint_folder, file)
        out_dir = os.path.dirname(f)
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir)
            except FileExistsError:
                pass
        self.sim.walkers.snapshot(f)

    def snapshot_trial_wavefunction(self, file="psit{core}_{n}.npz", save_stepnum = True):
        """Saves a snapshot of the walkers to a pickle

        :return:
        :rtype:
        """

        if self.sim.imp_samp is not None:
            psi = self.sim.imp_samp.psi
            if psi is not None:
                n = "" if not save_stepnum else self.sim.counter.step_num
                core = self.sim.world_rank
                if core == 0:
                    core = ""
                file = file.format(core=core, n=n)
                f = os.path.abspath(file)
                if file != f:
                    f = os.path.join(self.checkpoint_folder, file)
                out_dir = os.path.dirname(f)
                if not os.path.isdir(out_dir):
                    try:
                        os.makedirs(out_dir)
                    except FileExistsError:
                        pass
                np.save(f, psi)

    def save_wavefunction(self, wf, file = 'wavefunction{core}_{n}.npz'):
        """Save wavefunctions to a numpy binary

        :return:
        :rtype:
        """

        core = self.sim.world_rank
        if core == 0:
            core = ""
        n = self.sim.num_wavefunctions
        file = file.format(core=core, n=n)
        f = os.path.abspath(file)
        if file != f:
            file = os.path.join(self.wavefunctions_folder, file)
        wf_dir = os.path.dirname(file)
        if not os.path.isdir(wf_dir):
            try:
                os.makedirs(wf_dir)
            except FileExistsError:
                pass
        if not self.sim.dummied:
            self.log_print("Saving wavefunction to {}", file, verbosity=self.LogLevel.STEPS)
        np.savez(file, **wf)
        return file

    def snapshot_weights(self, file="full_weights{core}.npy"):
        """Saves a snapshot of the energies to a numpy binary

        :param file:
        :type file:
        :return:
        :rtype:
        """

        if self.sim.full_weights is not None:
            core = self.sim.world_rank
            if core == 0:
                core = ""
            file = file.format(core=core)

            f = os.path.abspath(file)
            if not os.path.isfile(f):
                if not os.path.isdir(self.output_folder):
                    try:
                        os.makedirs(self.output_folder)
                    except FileExistsError:
                        pass
                f = os.path.join(self.output_folder, file)

            np.save(f, np.array(self.sim.full_weights).astype("float"))
            return f

    def snapshot_full_energies(self, file="full_energies{core}.npy"):
        """Saves a snapshot of the energies to a numpy binary

        :param file:
        :type file:
        :return:
        :rtype:
        """

        if self.sim.full_energies is not None:
            core = self.sim.world_rank
            if core == 0:
                core = ""
            file = file.format(core=core)

            f = os.path.abspath(file)
            if not os.path.isfile(f):
                if not os.path.isdir(self.output_folder):
                    try:
                        os.makedirs(self.output_folder)
                    except FileExistsError:
                        pass

                f = os.path.join(self.output_folder, file)

            np.save(f, np.array(self.sim.full_energies).astype("float"))
            return f

    def snapshot_energies(self, file="energies{core}.npy"):
        """Saves a snapshot of the energies to a numpy binary

        :param file:
        :type file:
        :return:
        :rtype:
        """

        core = self.sim.world_rank
        if core == 0:
            core = ""
        file = file.format(core=core)

        f = os.path.abspath(file)
        if not os.path.isfile(f):
            if not os.path.isdir(self.output_folder):
                try:
                    os.makedirs(self.output_folder)
                except FileExistsError:
                    pass
            f = os.path.join(self.output_folder, file)

        np.save(f, np.array(self.sim.reference_potentials))
        return f

    def save_data(self, data, file, folder="checkpoints", save_stepnum=True):
        """
        Saves the passed data to a numpy file.
        Fills in template parameters in the file name.

        :param file:
        :type file:
        :return:
        :rtype:
        """

        n = "" if not save_stepnum else self.sim.counter.step_num

        core = self.sim.world_rank
        if core == 0:
            core = ""
        file = file.format(core=core, n=n)

        if folder == "checkpoints":
            folder = self.checkpoint_folder
        elif folder == "wavefunctions":
            folder = self.wavefunctions_folder
        elif folder == "output":
            folder = self.output_folder
        else:
            if os.path.abspath(folder) != folder:
                folder = os.path.join(self.output_folder, folder)

        f = os.path.abspath(file)
        if not os.path.isfile(f):
            if not os.path.isdir(folder):
                try:
                    os.makedirs(folder)
                except FileExistsError:
                    pass

            f = os.path.join(folder, file)

        np.save(f, np.asarray(data))
        return f

    def save_energies(self, energies, file="energies{core}_{n}.npy", save_stepnum=True):
        """
        Saves the given energies to a file

        :param energies:
        :type energies:
        :param file:
        :type file:
        :return:
        :rtype:
        """

        return self.save_data(energies, file, save_stepnum=save_stepnum)

    def save_coords(self, coords, file="coordinates{core}_{n}.npy", save_stepnum=True):
        """
        Saves the given energies to a file

        :param energies:
        :type energies:
        :param file:
        :type file:
        :return:
        :rtype:
        """

        return self.save_data(coords, file, save_stepnum=save_stepnum)

    def checkpoint(self, save_stepnum = True):
        # if not self.sim.dummied:
        self.log_print("Checkpointing simulation", verbosity=self.LogLevel.STEPS)
        # self.snapshot("checkpoint.pickle")
        self.snapshot_energies()
        self.snapshot_full_energies()
        self.snapshot_weights()
        # self.snapshot_params()
        self.snapshot_walkers(save_stepnum=save_stepnum)
        self.snapshot_trial_wavefunction(save_stepnum=save_stepnum)