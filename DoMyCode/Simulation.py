import os, numpy as np, time
from .WalkerSet import WalkerSet
from .ImportanceSampler import ImportanceSampler, ImportanceSamplerManager
from ..RynUtils import Logger, ParameterManager
from ..Dumpi import *
from ..PlzNumbers import PotentialManager

__all__ = [
    "Simulation",
    "SimulationParameters"
]

##############################################################################################################
#
#                                       Simulation Classes
#

class SimulationStepCounter:
    """
    A class that manages knowing what timestep it is and what that means we need to do
    """
    __props__ = [
        "step_num", "num_time_steps", "checkpoint_every",
        "equilibration_steps", "descendent_weight_every", "descendent_weighting_steps"
    ]
    def __init__(self,
                 sim,
                 step_num = 0,
                 num_time_steps = None,
                 checkpoint_at=None,
                 equilibration_steps=None,
                 descendent_weight_every = None,
                 descendent_weighting_steps = None
                 ):
        """
        :param step_num: the step number we're starting at
        :type step_num: int
        :param num_time_steps: the total number of timesteps we're running
        :type num_time_steps:
        :param checkpoint_every: how often to checkpoint the simulation
        :type checkpoint_every:
        :param equilibration_steps: the number of equilibration timesteps
        :type equilibration_steps: int
        :param descendent_weight_every: how often to calculate descendent weights
        :type descendent_weight_every: int
        :param descendent_weighting_steps: the number of steps taken in descendent weighting
        :type descendent_weighting_steps:
        """
        self.simulation = sim
        self.step_num = step_num
        self.num_time_steps = num_time_steps

        self._previous_checkpoint = step_num
        if isinstance(checkpoint_at, int):
            checkpoint_at = lambda s, c=checkpoint_at: s.step_num - s._previous_checkpoint >= c
        elif checkpoint_at is None:
            checkpoint_at = lambda *a: False
        self._checkpoint = checkpoint_at

        self._equilibrated = False
        if isinstance(equilibration_steps, int):
            equilibration_steps = lambda s, e=equilibration_steps: s.step_num > e  # classic lambda parameter binding
        self.equilibration_check = equilibration_steps

        self._dw_delay = descendent_weight_every
        self._dw_steps = descendent_weighting_steps
        self._last_dw_step = 0
        self._dw_initialized_step = None

    @property
    def checkpoint(self):
        checkpoint = self._checkpoint(self)
        if checkpoint:
            self._previous_checkpoint = self.step_num
        return checkpoint

    @property
    def equilibrated(self):
        if not self._equilibrated:
            self._equilibrated = self.equilibration_check(self)
        return self._equilibrated

    @property
    def done(self):
        return self.step_num >= self.num_time_steps

    def descendent_weighting_status(self):
        step = self.step_num
        if self._dw_initialized_step is None:
            status = "waiting"
        elif step - self._dw_initialized_step >= self._dw_steps:
            status = "complete"
            self._dw_initialized_step = None
        elif step - self._last_dw_step >= self._dw_delay:
            status = "beginning"
            self._dw_initialized_step = step
            self._dw_initialized_step = step
            self._last_dw_step = step
        else:
            status = "ongoing"
        return status


class SimulationTimer:
    """
    Super simple timer that does some formatting and shit
    """
    __props__ = [ "simulation_times" ]
    def __init__(self, simulation, simulation_times = None):
        self.simulation = simulation
        if simulation_times is None:
            simulation_times = []
        self.simulation_times = simulation_times
        self.start_time = None

    def start(self):
        self.start_time = time.time()
    def stop(self):
        if self.start_time is not None:
            self.simulation_times.append((self.start_time, time.time()))
            self.start_time = None
    @property
    def elapsed(self):
        if self.start_time is not None:
            return time.time() - self.start_time()
        else:
            return 0

class SimulationLogger:
    """
    A class for saving simulation data
    """

    LOG_BASIC = 1
    LOG_STATUS = 3
    LOG_STEPS = 5
    LOG_DATA = 6
    LOG_MPI = 7
    LOG_ALL = 10
    LOG_DEBUG = 100

    __props__ = [
        "output_folder",
        "write_wavefunctions",
        "save_snapshots"
        "log_file",
        "verbosity"
    ]
    def __init__(self,
                 simulation,
                 output_folder = None,
                 write_wavefunctions=True,
                 save_snapshots=None,
                 log_file = None,
                 verbosity = 100
                 ):
        self.sim = simulation
        self.output_folder = output_folder
        self.write_wavefunctions = write_wavefunctions
        self.save_snapshots = save_snapshots

        if output_folder is None:
            output_folder = os.path.join(os.path.abspath("dmc_data"), self.sim.name)
        self.output_folder = output_folder

        if log_file is None:
            log_file = os.path.join(self.output_folder, "log.txt")
        self.log_file = log_file

        self.verbosity = verbosity

        self.save_snapshots = save_snapshots

        self.logger = Logger(self.log_file, verbosity = self.verbosity)

    def log_print(self, *args, **kwargs):
        self.logger.log_print(*args, **kwargs)

    def snapshot(self, file="snapshot.pickle"):
        """Saves a snapshot of the simulation to file

        :param file:
        :type file:
        :return:
        :rtype:
        """
        raise NotImplementedError("Turns out pickle doesn't like this")

        # import pickle
        #
        # f = os.path.abspath(file)
        # if not os.path.isfile(f):
        #     if not os.path.isdir(self.output_folder):
        #         os.makedirs(self.output_folder)
        #     f = os.path.join(self.output_folder, file)
        # with open(f, "w+") as binary:
        #     pickle.dump(self, binary)

    def snapshot_params(self, file="params.pickle"):
        """Saves a snapshot of the params to a pickle

        :return:
        :rtype:
        """

        f = os.path.abspath(file)
        if not os.path.isfile(f):
            f = os.path.join(self.output_folder, file)
        out_dir = os.path.dirname(f)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.sim.params.serialize(f)

    def snapshot_walkers(self, file="walkers{}.npz", save_stepnum = True):
        """Saves a snapshot of the walkers to a pickle

        :return:
        :rtype:
        """

        file = file.format("" if save_stepnum else self.sim.counter.step_num)
        f = os.path.abspath(file)
        if not os.path.isfile(f):
            f = os.path.join(self.output_folder, file)
        out_dir = os.path.dirname(f)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.sim.walkers.snapshot(f)

    def save_wavefunction(self, test = True, file = 'wavefunction_{}.npz'):
        """Save wavefunctions to a numpy binary

        :return:
        :rtype:
        """
        if (not test) or self.write_wavefunctions:
            file = file.format(self.sim.num_wavefunctions)
            wf_dir = os.path.join(self.output_folder, "wavefunctions")
            if not os.path.isdir(wf_dir):
                os.makedirs(wf_dir)
            file = os.path.join(wf_dir, file)
            self.log_print("Saving wavefunction to {}", file, verbosity=self.LOG_STEPS)
            np.savez(file, **self.sim.wavefunctions[-1])
            return file

    def snapshot_energies(self, test = True, file="energies"):
        """Saves a snapshot of the energies to a numpy binary

        :param file:
        :type file:
        :return:
        :rtype:
        """

        f = os.path.abspath(file)
        if not os.path.isfile(f):
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder)
            f = os.path.join(self.output_folder, file)
        np.save(f, np.array(self.sim.reference_potentials))
        return f

    def checkpoint(self, test = True):
        if (not test) or self.save_snapshots:
            self.log_print("Checkpointing simulation", verbosity=self.LOG_STEPS)
            # self.snapshot("checkpoint.pickle")
            self.snapshot_energies()
            self.snapshot_params()
            self.snapshot_walkers(save_stepnum=self.save_snapshots == True)


class SimulationAnalyzer:
    __props__ = [ "zpe_averages" ]
    def __init__(self, simulation, zpe_averages = 1000):
        self.sim = simulation
        self.zpe_averages = zpe_averages

    @property
    def zpe(self):
        return self.get_zpe()

    def get_zpe(self, n=None):
        import itertools
        if n is None:
            n = self.zpe_averages
        if len(self.sim.reference_potentials) > n:
            vrefs = list(itertools.islice(self.sim.reference_potentials, len(self.sim.reference_potentials) - n, None, 1))
        else:
            vrefs = self.sim.reference_potentials
        # return Constants.convert(np.average(np.array(vrefs)), "wavenumbers", in_AU=False)

    class Plotter:
        _mpl_loaded = False

        @classmethod
        def load_mpl(cls):
            if not cls._mpl_loaded:
                import matplotlib as mpl
                # mpl.use('Agg')
                cls._mpl_loaded = True

        @classmethod
        def plot_vref(cls, sim):
            """

            :param sim:
            :type sim: Simulation
            :return:
            :rtype:
            """
            from ..RynUtils.Constants import Constants

            import matplotlib.pyplot as plt
            e = np.array(sim.reference_potentials)
            n = np.arange(len(e))
            fig, axes = plt.subplots()
            e = Constants.convert(e, 'wavenumbers', in_AU=False)
            axes.plot(n, e)
            # axes.set_ylim([-3000,3000])
            plt.show()

        @classmethod
        def plot_psi(cls, sim):
            """

            :param sim:
            :type sim: Simulation
            :return:
            :rtype:
            """
            # assumes 1D psi...
            import matplotlib.pyplot as plt
            w = sim.walkers
            fig, axes = plt.subplots()

            hist, bins = np.histogram(w.coords.flatten(), weights=(w.weights), bins=20, density=True)
            bins -= (bins[1] - bins[0]) / 2
            axes.plot(bins[:-1], hist)
            plt.show()

        @classmethod
        def plot_psi2(cls, sim):
            """

            :param sim:
            :type sim: Simulation
            :return:
            :rtype:
            """
            # assumes 1D psi...
            import matplotlib.pyplot as plt
            w = sim.walkers
            fig, axes = plt.subplots()
            coord, dw, ow = sim.wavefunctions[-1]
            coord = coord.flatten()

            hist, bins = np.histogram(coord, weights=dw, bins=20, density=True)
            bins -= (bins[1] - bins[0]) / 2
            axes.plot(bins[:-1], hist)
            plt.show()


class Simulation:
    """
    A DMC simulation class. Uses a number of subclasses to manage its methods
    """

    __props__ = [
        "name", "description",
        "walker_set", "time_step", "alpha",
        "potential", "steps_per_propagation",
        "mpi_manager", "importance_sampler"
    ]
    def __init__(self, params):
        """Initializes the simulation from the simulation parameters

        :param params: the parameters for the simulation
        :type params: SimulationParameters
        """
        self.params = params
        self.counter = SimulationStepCounter(self, **params.filter(SimulationStepCounter))
        self.timer = SimulationTimer(self, **params.filter(SimulationTimer))
        self.logger = SimulationLogger(self, **params.filter(SimulationLogger))
        self.analyzer = SimulationAnalyzer(self, **params.filter(SimulationAnalyzer))
        self.configure_simulation(**params.filter(Simulation))

    def configure_simulation(
            self,
            name = "dmc",
            description = "a dmc simulation",
            walker_set = None,
            time_step = None,
            alpha = None,
            potential = None,
            steps_per_propagation = None,
            mpi_manager = None,
            importance_sampler = None
            ):
        """

        :param name:
        :type name: str
        :param description:
        :type description: str
        :param walker_set:
        :type walker_set: WalkerSet
        :param time_step:
        :type time_step: int
        :param alpha:
        :type alpha: float
        :param potential:
        :type potential: str | Potential
        :param mpi_manager:
        :type mpi_manager: MPIManagerObject
        :param steps_per_propagation:
        :type steps_per_propagation: int
        :param importance_sampler:
        :type importance_sampler: ImportanceSampler
        :return:
        :rtype:
        """

        from collections import deque

        self.name = name
        self.description = description

        # basically we don't let it not use MPI...
        if mpi_manager is None:
            mpi_manager = MPIManager()
        elif isinstance(mpi_manager, str) and mpi_manager == "serial":
            mpi_manager = None

        if isinstance(walker_set, str):
            walker_set = WalkerSet.from_file(walker_set, mpi_manager=mpi_manager)
        elif isinstance(walker_set, dict):
            walker_set['mpi_manager']=mpi_manager
            walker_set = WalkerSet(**walker_set)

        self.walkers = walker_set if isinstance(walker_set, WalkerSet) else WalkerSet(walker_set)
        if isinstance(potential, str):
            potential = PotentialManager().load_potential(potential)
            potential.bind_atoms(walker_set.atoms)
        elif isinstance(potential, dict):
            pot = PotentialManager().load_potential(potential["name"])
            pot.bind_atoms(walker_set.atoms)
            if 'parameters' in potential:
                pot.bind_arguments(potential['parameters'])
            potential = pot
        potential.mpi_manager = mpi_manager
        self.potential = potential
        if alpha is None:
            alpha = 1.0 / (2.0 * time_step)
        self.alpha = alpha
        self.steps_per_propagation = steps_per_propagation

        self.reference_potentials = deque() # just a convenient data structure to push into

        D = 2.0 # not sure what this is supposed to be...?
        self.walkers.initialize(time_step, D)
        self.time_step = time_step

        self.wavefunctions = deque()
        self._num_wavefunctions = 0 # here so we can do things with save_wavefunction <- mattered in the past when I sometimes had deque(maxlength=1)

        self.mpi_manager = mpi_manager
        self.dummied = mpi_manager is None or mpi_manager.world_rank != 0

        if isinstance(importance_sampler, str):
            importance_sampler = ImportanceSamplerManager.load_sampler(importance_sampler)
        self.imp_samp = importance_sampler
        if self.imp_samp is not None:
            self.imp_samp.init_params(self.walkers.sigmas, self.time_step)

    def checkpoint(self, test = True):
        if (not self.dummied) and ((not test) or self.counter.checkpoint):
            self.logger.checkpoint()

    @classmethod
    def reload(cls,
               output_folder = None,
               params_file = "params.pickle",
               energies_file = 'energies.npy',
               walkers_file="walkers.npz"
               ):
        """Reloads a Simulation object from a director with specified params file

        :param core_dir:
        :type core_dir:
        :param params_file:
        :type params_file:
        """

        from collections import deque

        if not os.path.exists(os.path.abspath(params_file)):
            if output_folder is None:
                raise IOError("{}.{}: needs a 'params.py' file to reload from".format(
                    cls.__name__,
                    "reload"
                ))
            elif not os.path.isdir(output_folder):
                output_folder = os.path.join(os.path.abspath("dmc_data"), output_folder)
                params_file = os.path.join(output_folder, params_file)

        params = SimulationParameters.deserialize(params_file)
        output_folder = params.output_folder # this is what's gonna be fed into the simulation at the end of the day anyway...

        energies_file = os.path.join(output_folder, energies_file) if not os.path.isfile(energies_file) else energies_file
        energies = np.load(energies_file)

        walkers_file = os.path.join(output_folder, walkers_file) if not os.path.isfile(walkers_file) else walkers_file
        walkers = WalkerSet.load(walkers_file)
        params.update(walker_set = walkers)

        self = cls(params)
        self.reference_potentials = deque(energies)

        return self

    def log_print(self, *arg, **kwargs):
        self.logger.log_print(*arg, **kwargs)

    def _prop(self):
        while not self.counter.done:
            self.propagate()
    def run(self):
        """Runs the DMC until we've gone through the requested number of time steps, checkpoint-ing if there's a crash

        :return:
        :rtype:
        """
        try:
            self._prop()
        except:
            self.checkpoint(test=False)
            raise

    def propagate(self, nsteps = None):
        """Propagates the system forward n steps

        :param nsteps: number of steps to propagate for; None means automatic
        :type nsteps:
        :return:
        :rtype:
        """
        if nsteps is None:
            nsteps = self.steps_per_propagation

        if not self.dummied:
            self.log_print("Starting step {}", self.counter.step_num, verbosity=self.logger.LOG_STATUS)
            self.log_print("Moving coordinates {} steps", nsteps, verbosity=self.logger.LOG_STEPS)
            coord_sets = self.walkers.displace(nsteps, importance_sampler=self.imp_samp)
            self.log_print("Computing potential energy", verbosity=self.logger.LOG_STATUS)
            start = time.time()
            energies = self.potential(coord_sets)
            if self.imp_samp is not None:
                imp = self.imp_samp #type: ImportanceSampler
                energies += imp.local_kin(coord_sets)
            end = time.time()
            self.log_print("    took {}s", end-start, verbosity=self.logger.LOG_STATUS)
            self.counter.step_num += nsteps
            self.log_print("Updating walker weights", verbosity=self.logger.LOG_STEPS)
            weights = self.update_weights(energies, self.walkers.weights)
            self.walkers.weights = weights
            self.log_print("Branching", verbosity=self.logger.LOG_STEPS)
            self.branch()
            self.log_print("Applying descendent weighting", verbosity=self.logger.LOG_STEPS)
            self.descendent_weight()
            self.checkpoint()
            if self.logger.verbosity >= self.logger.LOG_STATUS:
                # we do the check here so as to not waste time computing ZPE... even though that waste is effectively 0
                self.log_print("Zero-point Energy: {}", self.analyzer.zpe, verbosity=self.logger.LOG_STATUS)
            self.log_print("Runtime: {}s", round(self.timer.elapsed), verbosity=self.logger.LOG_STATUS)
        else:
            self.log_print(
                "    computing potential energy on core {}",
                self.mpi_manager.world_rank,
                verbosity=self.logger.LOG_MPI
            )
            try:
                walk = self._dummy_walkers
            except AttributeError:
                self._dummy_walkers = np.broadcast_to(self.walkers.coords, (nsteps,) + self.walkers.coords.shape).copy()
                walk = self._dummy_walkers
            self.potential(walk)
            self.counter.step_num += nsteps

    def _compute_vref(self, energies, weights):
        """Takes a single set of energies and weights and computes the average potential

        :param energies: single set of energies
        :type energies:
        :param weights: single set of weights
        :type weights:
        :return:
        :rtype: float
        """

        energy_threshold = 10.8 # cutoff above which potential is really an error
        pick_spec = energies < energy_threshold
        e_pick = energies[pick_spec]
        w_pick = weights[pick_spec]
        Vbar = np.average(e_pick, weights=w_pick, axis = 0)
        num_walkers = len(weights)
        correction=np.sum(weights-np.ones(num_walkers), axis = 0)/num_walkers
        vref = Vbar - (self.alpha * correction)
        return vref

    def update_weights(self, energies, weights):
        """Iteratively updates the weights over a set of vectors of energies

        :param energies:
        :type energies: np.ndarray
        :param weights:
        :type weights: np.ndarray
        :return:
        :rtype: np.ndarray
        """
        for e in energies: # this is basically a reduce call, but there's no real reason not to keep it like this
            self.log_print("Energies: {}", e, verbosity=self.logger.LOG_DATA)
            Vref = self._compute_vref(e, weights)
            self.reference_potentials.append(Vref) # a constant time operation
            new_wts = np.nan_to_num(np.exp(-1.0 * (e - Vref) * self.time_step))
            weights *= new_wts
            self.log_print("Weights: {}", weights, verbosity=self.logger.LOG_DATA)
        return weights

    def branch(self):
        """Handles branching in the system.

        :return:
        :rtype:
        """

        # this is the only place where we actually reach into the walkers...
        weights = self.walkers.weights
        walkers = self.walkers.coords
        parents = self.walkers.parents
        threshold = 1.0 / self.walkers.num_walkers

        eliminated_walkers = np.argwhere(weights < threshold).flatten()
        self.log_print('Walkers being removed: {}', len(eliminated_walkers), verbosity=self.logger.LOG_STATUS)
        self.log_print('Max weight in ensemble: {}', np.amax(weights), verbosity=self.logger.LOG_STATUS)

        for dying in eliminated_walkers:  # gotta do it iteratively to get the max_weight_walker right...
            cloning = np.argmax(weights)
            # print(cloning)
            parents[dying] = parents[cloning]
            walkers[dying] = walkers[cloning]
            weights[dying] = weights[cloning] / 2.0
            weights[cloning] /= 2.0

    def descendent_weight(self):
        """Calls into the walker descendent weighting if the timing is right

        :return:
        :rtype:
        """
        if self.counter.equilibrated:
            status = self.counter.descendent_weighting_status()
            if status == "complete":
                self.log_print("Collecting descendent weights", verbosity=self.logger.LOG_STATUS)
                dw = self.walkers.descendent_weight() # not sure where I want to cache these...
                self.wavefunctions.append(dw)
                self._num_wavefunctions += 1
                self.logger.save_wavefunction()
            elif status == "beginning":
                self.log_print("Starting descendent weighting propagation", verbosity=self.logger.LOG_STATUS)
                self.walkers._setup_dw()

class SimulationParameters(ParameterManager):
    """
    A parameters class that manages the data for a DMC simulation
    """
    def __init__(self, **params):
        """Sets up all the necessary simulation data to run a DMC

        :param name: name to be used when storing file data
        :type name: str
        :param description: long description which isn't used for anything
        :type description: str
        :param walker_set: the WalkerSet object that handles all the pure walker activities in the simulation
        :type walker_set: WalkerSet | dict
        :param time_step: the size of the timestep to use throughout the calculation
        :type time_step: float
        :param steps_per_propagation: the number of steps to move over before branching in a propagate call
        :type steps_per_propagation: int
        :param num_time_steps: the total number of time steps the simulation should run for (initially)
        :type num_time_steps: int
        :param alpha: used in finding the branching correction to the reference potential
        :type alpha: float
        :param potential: the function that will take a set of atoms and sets of configurations and spit back out potential value
        :type potential: function or callable
        :param descendent_weighting: the number of steps before descendent weighting and the number of steps to go before saving
        :type descendent_weighting: (int, int)
        :param log_file: the file to write log stuff to
        :type log_file: str or stream or other file-like-object
        :param output_folder: the folder to write all data stuff to
        :type output_folder: str
        :param equilibration: the number of timesteps or method to determine equilibration
        :type equilibration: int or callable
        :param write_wavefunctions: whether or not to write wavefunctions to file after descedent weighting
        :type write_wavefunctions: bool
        :param checkpoint_at: the number of timesteps to progress before checkpointing (None means never)
        :type checkpoint_at: int or None
        :param verbosity: the verbosity level for log printing
        :type verbosity: int
        :param zpe_averages: the number of steps to average the ZPE over
        :type zpe_averages: int
        :param dummied: whether or not to just use for potential calls (exists for hooking into MPI and parallel methods)
        :type dummied: bool
        :param world_rank: the world_rank of the processor in an MPI call
        :type world_rank: int
        """
        super().__init__(params)

    def serialize(self, simulation, file, mode = None):
        # we need to update the step_num argument, delete the WalkerSet object, and maybe the potential?
        self.update(step_num = simulation.counter.step_num, walker_set = None)
