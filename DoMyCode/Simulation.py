"""
Defines the core DMC simulation
"""

import os, numpy as np, time
from .WalkerSet import WalkerSet
from .ImportanceSampler import ImportanceSampler
from .ImportanceSamplerManager import ImportanceSamplerManager
from ..RynUtils import ParameterManager, CLoader
from ..Dumpi import *
from ..PlzNumbers import PotentialManager, Potential
from .SimulationUtils import *

__all__ = [
    "Simulation",
    "SimulationParameters"
]

##############################################################################################################
#
#                                       Simulation Classes
#

class Simulation:
    """
    A DMC simulation class. Uses a number of subclasses to manage its methods
    """

    __props__ = [
        "name", "description",
        "walker_set", "time_step", "alpha",
        "potential", "steps_per_propagation",
        "mpi_manager", "importance_sampler",
        "num_wavefunctions", "atomic_units",
        "ignore_errors",
        "branching_threshold", "min_potential_threshold",
        "max_weight_threshold",
        "parallelize_diffusion",
        "branch_on_cores", "branch_on_steps",
        "random_seed",
        "pre_run_script", "post_run_script",
        "save_all_evaluations"
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
        self._lib = None

    def configure_simulation(
            self,
            name = "dmc",
            description = "a dmc simulation",
            walker_set = None,
            time_step = 0,
            alpha = None,
            potential = None,
            atomic_units = False,
            steps_per_propagation = None,
            mpi_manager = True,
            importance_sampler = None,
            num_wavefunctions = 0,
            ignore_errors = False,
            branching_threshold = 1.0,
            energy_error_value = 10e8,
            max_weight_threshold = None,
            min_potential_threshold= None,
            branch_on_steps = False,
            parallelize_diffusion=True,
            branch_on_cores = False,
            random_seed = None,
            pre_run_script=None,
            post_run_script=None,
            save_all_evaluations=False
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

        if mpi_manager is True:
            mpi_manager = MPIManager()
        # elif isinstance(mpi_manager, str) and mpi_manager == "serial":
        #     mpi_manager = None
        self.atomic_units = atomic_units
        self.ignore_errors = ignore_errors

        # branching flags
        self.branch_on_cores = branch_on_cores
        self.branching_threshold = branching_threshold
        self.min_potential_threshold = min_potential_threshold
        self.max_weight_threshold = max_weight_threshold
        self.branch_on_steps = branch_on_steps
        self.energy_error_value = energy_error_value
        self.parallelize_diffusion = parallelize_diffusion

        if alpha is None:
            alpha = 1.0 / (2.0 * time_step)
        self.alpha = alpha
        self.steps_per_propagation = steps_per_propagation

        self.reference_potentials = deque() # just a convenient data structure to push into

        # if we want to log everything we can
        if self.logger.verb_int >= self.logger.LogLevel.ALL.value:
            self.full_energies = deque()
            self.full_weights = deque()
        else:
            self.full_energies = None
            self.full_weights = None

        self.time_step = time_step

        self.wavefunctions = deque()
        self.num_wavefunctions = num_wavefunctions

        # Load all core objects

        # MPI Manager
        self.mpi_manager = mpi_manager
        try:
            self.world_rank = 0 if mpi_manager is None else mpi_manager.world_rank
            # self.log_print("Testing MPI validity on {}: {}".format(self.world_rank, self.mpi_manager.test()))
        except:
            import traceback as tb
            self.log_print("Error Occurred in configuring MPI\n  {}", tb.format_exc(), self.logger.LogLevel.MPI)
            mpi_manager.abort()
            raise
        self.dummied = self.world_rank != 0

        # Random Seed
        if random_seed is not None:
            if self.dummied:
                random_seed += self.world_rank
            np.random.seed(random_seed)
        self.random_seed = random_seed

        ### Walkers
        if isinstance(walker_set, str):
            walker_set = WalkerSet.from_file(walker_set, mpi_manager=mpi_manager)
        elif isinstance(walker_set, dict):
            walker_set['mpi_manager']=mpi_manager
            walker_set = WalkerSet(**walker_set)

        self.walkers = walker_set if isinstance(walker_set, WalkerSet) else WalkerSet(walker_set)
        self.walkers.initialize(time_step)

        # define a set of masks for tracking info about walkers
        self.masks = WalkerMask(self.walkers)

        ### Potential
        if isinstance(potential, str):
            potential = PotentialManager().load_potential(potential)
            potential.bind_atoms(walker_set.atoms)
        elif isinstance(potential, dict):
            pot = PotentialManager().load_potential(potential["name"])
            pot.bind_atoms(walker_set.atoms)
            if 'parameters' in potential:
                pot.bind_arguments(potential['parameters'])
            potential = pot
        self.potential = potential

        ### Importance Sampler
        if isinstance(importance_sampler, dict):
            params = importance_sampler["parameters"]
            importance_sampler = importance_sampler["name"]
        else:
            params = None

        if isinstance(importance_sampler, str):
            importance_sampler = ImportanceSamplerManager().load_sampler(importance_sampler)

        self.imp_samp = importance_sampler
        if self.imp_samp is not None:
            # if self.parallelize_diffusion:
            #     raise ValueError("Can't both do parallel diffusion & importance sampling, just by the way this is currently implemented")
            # else:
            #     parallelize_diffusion = False
            if params is None:
                params = ()
            if not parallelize_diffusion:
                mpi = self.mpi_manager
            else:
                mpi = None
            self.imp_samp.init_params(
                self.walkers.sigmas,
                self.time_step,
                mpi,
                self.walkers.atoms,
                *params,
                atomic_units=self.atomic_units
            )

        if not self.parallelize_diffusion:
            potential.mpi_manager = self.mpi_manager
        else:
            potential.mpi_manager = None

        self.save_all_evaluations = save_all_evaluations

        self.pre_run_script=pre_run_script
        self.post_run_script=post_run_script

    @property
    def config_string(self):
        from ..Interface import RynLib
        header = "RynLib DMC SIMULATION (version {}): {}\n{}\n".format(
            RynLib.VERSION_NUMBER,
            self.name,
            self.description
        )
        logger_props = "\n".join([
            "{}: {}".format(k, getattr(self.logger, k)) for k in [
                'verbosity'
            ]
        ])
        sim_props = "\n".join([
            "{}: {}".format(k, getattr(self, k)) for k in [
                "potential",
                "imp_samp",
                'mpi_manager',
                'branching_threshold',
                'branch_on_steps',
                'branch_on_cores',
                'parallelize_diffusion',
                'random_seed',
                'pre_run_script',
                'post_run_script',
                'atomic_units',
                'ignore_errors',
                'min_potential_threshold',
                'max_weight_threshold',
                'energy_error_value'
            ]
        ])
        walk_props = "\n".join([
            "{}: {}".format(k, getattr(self.walkers, k)) for k in [
                'num_walkers',
                'atoms',
                'masses',
                'sigmas'
            ]
        ])
        params_props = "\n".join([
            "{}: {}".format(k, v) for k, v in self.params.items()
        ])
        c_string = "\n".join([header, logger_props, sim_props, walk_props, "-"*50, params_props])
        return c_string

    def checkpoint(self, test = True):
        can_check = self.counter.checkpoint
        # self.log_print("Checkpoint? {}", can_check, verbosity=self.logger.LogLevel.STATUS)
        if (not test) or can_check:
            self.logger.checkpoint()

    def garbage_collect(self, test = True):
        import gc

        can_gc = self.counter.checkpoint
        # self.log_print("Checkpoint? {}", can_check, verbosity=self.logger.LogLevel.STATUS)
        if (not test) or can_gc:
            gc.collect()

    # @property
    # def checkpoint_params(self):
    #     # anything that isn't stored in config.py
    #     return {
    #         'step_num':self.counter.step_num
    #     }

    def reload(self,
               # params_file = "checkpoint.json",
               energies_file = 'energies.npy',
               walkers_file="walkers_{n}.npz",
               full_weights_file="full_weights.npy",
               full_energies_file='full_energies.npy'
               ):
        """Reloads the core data in a Simulation object from a checkpoint file

        :param core_dir:
        :type core_dir:
        :param params_file:
        :type params_file:
        """

        from collections import deque

        # if not os.path.exists(os.path.abspath(params_file)):
        #     if output_folder is None:
        #         raise IOError("{}.{}: needs a 'params.py' file to reload from".format(
        #             cls.__name__,
        #             "reload"
        #         ))
        #     elif not os.path.isdir(output_folder):
        #         output_folder = os.path.join(os.path.abspath("dmc_data"), output_folder)
        #         params_file = os.path.join(output_folder, params_file)
        # params = SimulationParameters.deserialize(params_file)

        output_folder = self.logger.output_folder

        # LOAD ENERGIES
        energies_file = os.path.join(output_folder, energies_file) if not os.path.isfile(energies_file) else energies_file
        energies = np.load(energies_file) # this implicitly tells us how many steps we made it forward
        self.reference_potentials = deque(energies)

        # LOAD WALKERS
        step_num = len(energies)
        self.counter.step_num = step_num

        if not self.dummied:
            if not os.path.isfile(walkers_file):
                walkers_file = os.path.join(self.logger.checkpoint_folder, walkers_file.format(n=step_num))
            walkers = WalkerSet.load(walkers_file)
            self.walkers=walkers

            # LOAD WAVEFUNCTIONS
            wavefunctions_directory=self.logger.wavefunctions_folder
            if not os.path.isdir(wavefunctions_directory):
                wavefunctions_directory = os.path.join(output_folder, wavefunctions_directory)
            wfs = []
            if os.path.isdir(wavefunctions_directory):
                for f in os.listdir(wavefunctions_directory):
                    if f.endswith(".npz"):
                        wfs.append(
                            ( int(f.split("_")[-1].split(".")[0]), np.load(os.path.join(wavefunctions_directory, f)))
                        )
                    wfs = sorted(wfs, key=lambda a:a[0])
            self.wavefunctions = deque(wfs)
            self.num_wavefunctions = len(self.wavefunctions)

            # LOAD FULL WEIGHTS (if saved)
            if not os.path.isfile(full_weights_file):
                full_weights_file = os.path.join(output_folder, full_weights_file)
            if os.path.isfile(full_weights_file):
                self.full_weights = deque(np.load(full_weights_file))

            # LOAD FULL ENERGIES (if saved)
            if not os.path.isfile(full_energies_file):
                full_energies_file = os.path.join(output_folder, full_energies_file)
            if os.path.isfile(full_energies_file):
                self.full_energies = deque(np.load(full_energies_file))

        return self

    def log_print(self, *arg, allow_dummy=False, **kwargs):
        if allow_dummy or (not self.dummied):
            self.logger.log_print(*arg, **kwargs)

    def _prop(self):
        while not self.counter.done:
            self.propagate()
    def run(self):
        """Runs the DMC until we've gone through the requested number of time steps, checkpoint-ing if there's a crash

        :return:
        :rtype:
        """

        if not self.counter.done:

            if self.pre_run_script is not None:
                with open(self.pre_run_script) as script:
                    blob = script.read()
                code_blob = compile(blob, self.pre_run_script, 'exec')
                exec(code_blob, globals(), {'simulation': self})
            try:
                self.log_print(self.config_string)
                self.log_print("-"*50)
                self.log_print("Starting simulation")
                if self.mpi_manager is not None:
                    # self.log_print("waiting for friends", verbosity=self.logger.LogLevel.STATUS)
                    self.mpi_manager.wait()
                self.timer.start()

                self._prop()

            except Exception as e:
                import traceback as tb
                if not self.dummied:
                    self.log_print("Error Occurred\n  {}", tb.format_exc().replace("\n", "\n  "), verbosity=self.logger.LogLevel.STATUS)
                else:
                    self.log_print("Error Occurred on core {}\n  {}",
                                   self.world_rank, tb.format_exc().replace("\n", "\n  "),
                                   allow_dummy=True,
                                   verbosity=self.logger.LogLevel.STATUS
                                   )
                if not self.ignore_errors:
                    if self.mpi_manager is not None:
                        self.mpi_manager.abort()
                    raise
            finally:
                self.log_print("Ending simulation")
                self.checkpoint(test=False)
                if self.mpi_manager is not None:
                    self.mpi_manager.finalize_MPI()
                if isinstance(self.potential, Potential):
                    self.potential.clean_up()
                if isinstance(self.imp_samp, ImportanceSampler):
                    self.imp_samp.clean_up()
                self.timer.stop()

        if self.post_run_script is not None:
            with open(self.post_run_script) as script:
                blob = script.read()
            code_blob = compile(blob, self.post_run_script, 'exec')
            exec(code_blob, globals(), {'simulation': self})

    @classmethod
    def load_lib(cls):
        loader = CLoader("DoMyCode",
                         os.path.dirname(os.path.abspath(__file__)),
                         extra_compile_args=['-fopenmp', '-std=c++11'],
                         source_files=["DoMyCode.cpp", "PyAllUp.cpp"]
                         )
        return loader.load()
    @classmethod
    def reload_lib(cls):
        try:
            os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DoMyCode.so"))
        except OSError:
            pass
        cls.load_lib()
    @property
    def lib(self):
        if self._lib is None:
            self._lib = self.load_lib()
        return self._lib

    def _evaluate_potential(self, coord_sets):
        energies = self.potential(coord_sets)
        if self.imp_samp is not None:
            imp = self.imp_samp  # type: ImportanceSampler
            ke = imp.local_kin(coord_sets)
            if not self.dummied:
                self.log_print("    Local KE: Min {} | Max {} | Mean {}",
                               np.min(ke), np.max(ke), np.average(ke),
                               verbosity=self.logger.LogLevel.DATA
                               )
            if energies is not None and ke is not None:
                energies += ke
        return energies
    def apply_branching(self, energies):
        if not self.branch_on_cores:
            self.log_print("Updating walker weights", verbosity=self.logger.LogLevel.STEPS)
        # raise Exception(energies.shape)
        weights = self.update_weights(energies, self.walkers.weights)
        self.walkers.weights = weights
        if not self.branch_on_cores:
            self.log_print("Branching", verbosity=self.logger.LogLevel.STEPS)
            start = time.time()
        energies = self.branch(energies)
        if not self.branch_on_cores:
            end = time.time()
            self.log_print("  took {}s", end - start, verbosity=self.logger.LogLevel.STEPS)
        return self.walkers.coords, self.walkers.weights, energies
    def evaluate_potential_and_branch(self, nsteps):
        import sys
        def log_refcounts(names, objs):
            self.log_print(
                "\n  ".join(["Ref Counts:"]+
                    ["{}({}) {}"]*len(names)
                ),
                *[x for n,o in zip(names, objs) for x in (n, id(o), sys.getrefcount(o))],
                allow_dummy=True
            )

        if not self.parallelize_diffusion or self.mpi_manager is None:
            self.log_print("Moving coordinates {} steps", nsteps, verbosity=self.logger.LogLevel.STEPS)
            start = time.time()
            coord_sets, rejections = self.walkers.displace(nsteps, importance_sampler=self.imp_samp, atomic_units=self.atomic_units)
            end = time.time()
            if rejections is not None:
                for r in rejections:
                    self.log_print("    metropolis rejected {} walkers", r, verbosity=self.logger.LogLevel.STATUS)
            self.log_print("    took {}s", end - start, verbosity=self.logger.LogLevel.STATUS)
            self.log_print("Computing potential energy", verbosity=self.logger.LogLevel.STATUS)
            start = time.time()

            if self.save_all_evaluations:
                self.logger.save_coords(coord_sets)
            energies = self._evaluate_potential(coord_sets)
            if self.save_all_evaluations:
                self.logger.save_energies(energies)

            end = time.time()
            self.log_print("    took {}s", end - start, verbosity=self.logger.LogLevel.STATUS)
            if not self.dummied:
                coords, weights, energies = self.apply_branching(energies)
            else:
                energies = weights = None
        else:
            send_walkers = self.lib.sendFriends
            get_results = self.lib.getFriendsAndPoots
            # self.lib.DEBUG_LEVEL = 100

            self.log_print("Moving coordinates {} steps", nsteps, verbosity=self.logger.LogLevel.STEPS)
            start = time.time()

            coords = np.ascontiguousarray(self.walkers.coords).astype('float')
            small_walkers = send_walkers(coords, self.mpi_manager)

            self.walkers.coords = small_walkers
            coord_sets, rejections = self.walkers.get_displaced_coords(nsteps, importance_sampler=self.imp_samp, atomic_units=self.atomic_units)
            if rejections is not None:
                for r in rejections:
                    self.log_print("    metropolis rejected {} walkers", r, verbosity=self.logger.LogLevel.STATUS)

            end = time.time()
            self.log_print("    took {}s", end - start, verbosity=self.logger.LogLevel.STATUS)
            start = end
            self.log_print("Computing potential energy", coord_sets.shape, verbosity=self.logger.LogLevel.STATUS)

            if self.save_all_evaluations:
                self.logger.save_coords(coord_sets)
            energies = self._evaluate_potential(coord_sets)
            if self.save_all_evaluations:
                self.logger.save_energies(energies)

            if not self.dummied:
                end = time.time()
                self.log_print("    took {}s", end - start, verbosity=self.logger.LogLevel.STATUS)

            coords = np.ascontiguousarray(coord_sets[-1]).astype('float')
            energies = np.ascontiguousarray(energies).astype('float')

            if self.branch_on_cores:
                self.mpi_manager.wait()
                self.log_print("Branching", verbosity=self.logger.LogLevel.STATUS)
                self.mpi_manager.wait()
                self.log_print("   branching on core {}", self.mpi_manager.world_rank,
                               allow_dummy=True,
                               verbosity=self.logger.LogLevel.MPI
                               )
                coord_sets, weights, energies = self.apply_branching(energies)
            else:
                weights = None

            if not self.dummied:
                self.log_print("    getting energies/coords off cores")
            res = get_results(coords, energies, weights, self.mpi_manager)

            if res is None:
                self.walkers.coords = coords
                energies = None
            elif len(res) == 3: #branch_on_cores
                coords, energies, weights = res
                self.walkers.coords = coords
                self.walkers.weights = weights
                # we now have to take the energies arrays and stitch them together to look like the coords ;_;
                world_size = int(energies.shape[0] / nsteps)
                walkers_per_core = energies.shape[1]
                real_energies = np.empty((nsteps, walkers_per_core * world_size), dtype=energies.dtype)
                for i in range(nsteps):
                    for j in range(world_size):
                        real_energies[i, walkers_per_core * j:walkers_per_core * (j + 1)] = energies[i + nsteps * j]
                energies = real_energies
            elif len(res) == 2: # implicitly means not self.branch_on_cores
                coords, energies = res
                # we now have to take the energies arrays and stitch them together to look like the coords ;_;
                world_size = int(energies.shape[0] / nsteps)
                walkers_per_core = energies.shape[1]
                real_energies = np.empty((nsteps,  walkers_per_core * world_size), dtype=energies.dtype)
                for i in range(nsteps):
                    for j in range(world_size):
                        real_energies[i, walkers_per_core*j:walkers_per_core*(j+1)] = energies[i+nsteps*j]
                energies = real_energies

                self.walkers.coords = coords
                coords, weights, energies = self.apply_branching(energies)

        return energies, weights

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
            self.log_print("Starting step {}", self.counter.step_num, verbosity=self.logger.LogLevel.STATUS)
            energies, weights = self.evaluate_potential_and_branch(nsteps)
            self.counter.increment(nsteps)
            self.log_print("Applying descendent weighting", verbosity=self.logger.LogLevel.STEPS)
            self.descendent_weight()
            if self.full_energies is not None:
                self.full_energies.append(energies)
            if self.full_weights is not None:
                self.full_weights.append(weights)

            self.log_print("Pre-checkpointing?", verbosity=self.logger.LogLevel.STEPS)
            self.checkpoint()
            # self.garbage_collect()
            if self.logger.verb_int >= self.logger.LogLevel.STATUS.value:
                # we do the check here so as to not waste time computing ZPE... even though that waste is effectively 0
                self.log_print("Average Energy: {}", self.reference_potentials[-1], verbosity=self.logger.LogLevel.STATUS)
                self.log_print("Zero-point Energy: {}", self.analyzer.zpe, verbosity=self.logger.LogLevel.STATUS)
            self.log_print("Runtime: {}s", round(self.timer.elapsed), verbosity=self.logger.LogLevel.STATUS)
            self.log_print("-" * 50, verbosity=self.logger.LogLevel.STATUS)
        else:
            energies, weights = self.evaluate_potential_and_branch(nsteps)
            self.counter.increment(nsteps)
            # self.garbage_collect()

        if self.mpi_manager is not None and not self.counter.done: # just in case the subsidiary and main cores get off
            # self.log_print("waiting for friends", verbosity=self.logger.LogLevel.STATUS)
            self.mpi_manager.wait()

    def _compute_vref(self, energies, weights):
        """
        Takes a single set of energies and weights and computes the average potential

        :param energies: single set of energies
        :type energies:
        :param weights: single set of weights
        :type weights:
        :return:
        :rtype: float
        """

        energy_threshold = self.energy_error_value # cutoff above which potential is really an error
        pick_spec = np.logical_and(energies <= energy_threshold, weights > 0.0)
        e_pick = energies[pick_spec]
        w_pick = weights[pick_spec]
        if len(w_pick) == 0:
            self.ignore_errors = False
            raise ValueError("All walkers have a weight of zero")
        Vbar = np.average(e_pick, weights=w_pick, axis = 0)
        # we assume here that all walkers were initialized with a weight of one
        # we also have to account for the fact that we zero out a bunch of weights to kill the walkers...
        num_walkers = len(weights)
        zeros = np.sum(weights <= 0.0)
        correction=(np.sum(w_pick-np.ones(len(w_pick)), axis = 0) - zeros)/num_walkers
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

        self.masks.reset()

        if self.branch_on_steps:
            on_step = True
            threshold = self.branching_threshold / self.walkers.num_walkers
        else:
            on_step = False
            threshold = None

        if self.min_potential_threshold is not None:
            e_min = self.min_potential_threshold
        else:
            e_min = None

        if self.max_weight_threshold is not None:
            max_w = self.max_weight_threshold
        else:
            max_w = None

        if e_min is not None:
            self.log_print("    Applying min energy threshold of {}",
                          e_min,
                          verbosity=self.logger.LogLevel.DATA
                          )
        if threshold is not None:
            self.log_print("    Applying weight threshold of {}",
                          threshold,
                          verbosity=self.logger.LogLevel.DATA
                          )
        if max_w is not None:
            self.log_print("    Applying max weight threshold of {}",
                          max_w,
                          verbosity=self.logger.LogLevel.DATA
                          )

        nsteps = len(energies)
        for i, e in enumerate(energies):
            self.log_print("    Energy: Min {} | Max {} | Mean {}",
                           np.min(e), np.max(e), np.average(e),
                           verbosity=self.logger.LogLevel.DATA
                           )

            # Applying energy threshold
            if e_min is not None:
                e_spec = e < e_min
                num_bad = np.sum(e_spec)
                if num_bad > 0:
                    self.log_print("    Dropping {} walkers below energy threshold",
                                   num_bad,
                                   verbosity=self.logger.LogLevel.DATA
                                   )
                    self.masks[w_spec] = self.masks.EnergyTooLow
                    e[e_spec] = self.energy_error_value
                    weights[e_spec] = -1.0

            # Computing new weights
            Vref = self._compute_vref(e, weights)
            self.reference_potentials.append(Vref) # a constant time operation
            new_wts = np.nan_to_num(np.exp(-1.0 * (e - Vref) * self.time_step))
            weights *= new_wts

            # Applying max-weight threshold
            if on_step or i == nsteps - 1:
                if max_w is not None:
                    w_spec = weights > max_w
                    num_bad = np.sum(w_spec)
                    if num_bad > 0:
                        self.log_print("    Marking {} walkers above max weight threshold for branching",
                                       num_bad,
                                       verbosity=self.logger.LogLevel.DATA
                                       )
                        self.masks[w_spec] = self.masks.WeightTooHigh
                        # e[w_spec] = self.energy_error_value
                        # weights[w_spec] = -1.0

            # Applying min-weight threshold
            if on_step or i == nsteps - 1:
                if threshold is not None:
                    w_spec = np.logical_and(weights < threshold, weights > 0.0)
                    num_bad = np.sum(w_spec)
                    if num_bad > 0:
                        self.log_print("    Dropping {} walkers below min weight threshold",
                                       num_bad,
                                       verbosity=self.logger.LogLevel.DATA
                                       )
                        self.masks[w_spec] = self.masks.WeightTooLow
                        e[w_spec] = self.energy_error_value
                        weights[w_spec] = -1.0

            self.log_print(
                "    Weight: Min {} | Max {} | Mean {}",
                np.min(weights), np.max(weights), np.average(weights),
                verbosity=self.logger.LogLevel.DATA
            )

        return weights

    @staticmethod
    def chop_weights(eliminated_walkers, weights, parents, walkers, energies):
        n_elims = len(eliminated_walkers)
        max_inds = np.argpartition(weights, -n_elims)[-n_elims:]
        max_weights = weights[max_inds].copy()
        max_sort = np.flip(np.argsort(max_weights))
        max_inds = max_inds[max_sort]
        max_weights = max_weights[max_sort]
        if max_weights[0] / 2 < max_weights[-1]:
            # we can circumvent any work
            cloning = max_inds
            dying = eliminated_walkers
            eliminated_walkers = []
        else:
            mean_guys = np.sum(max_weights[0] / 2 > max_weights)  # this is the number of walkers we actually need to be careful with
            cloning = max_inds[:-mean_guys]
            dying = eliminated_walkers[:-mean_guys]
            eliminated_walkers = eliminated_walkers[-mean_guys:]
        weights[cloning] /= 2.0
        weights[dying] = weights[cloning]
        parents[dying] = parents[cloning]
        walkers[dying] = walkers[cloning]
        if energies.ndim > 1:
            energies[-1, dying] = energies[-1, cloning]
        else:
            energies[dying] = energies[cloning]
        return eliminated_walkers

    def branch(self, energies, max_its=10):
        """
        Handles branching in the system.
        """

        # this is the only place where we actually reach into the walkers...
        weights = self.walkers.weights
        walkers = self.walkers.coords
        parents = self.walkers.parents
        threshold = self.branching_threshold / self.walkers.num_walkers

        # bond_lengths = np.linalg.norm(walkers[:, 1] - walkers[:, 0], axis=1)
        # self.log_print('Bond Lengths: Min {} | Max {} | Mean {} | Vib. Avg {}',
        #                np.min(bond_lengths), np.max(bond_lengths),
        #                np.average(bond_lengths), np.average(bond_lengths, weights=weights)
        #                )

        eliminated_walkers = np.argwhere(weights < threshold).flatten()
        num_elim = len(eliminated_walkers)
        branch_energies = energies[-1, eliminated_walkers] if num_elim > 0 else np.array([np.nan])
        branch_weights = weights[eliminated_walkers] if num_elim > 0 else np.array([np.nan])
        self.log_print(
            '\n    '.join([
                'Walkers being removed: {}',
                'Threshold: {}',
                'Energy: Min {} | Max {} | Mean {}',
                'Weight: Min {} | Max {} | Mean {}'
                ]),
            num_elim, threshold,
            np.min(branch_energies), np.max(branch_energies), np.average(branch_energies),
            np.min(branch_weights), np.max(branch_weights), np.average(branch_weights),
            verbosity=self.logger.LogLevel.STATUS
            )

        num_its = 0
        while len(eliminated_walkers) > 0 and num_its < max_its:
            eliminated_walkers = self.chop_weights(eliminated_walkers, weights, parents, walkers, energies)
            num_its += 1

        # WalkerMasks.where basicall just returns a np.where statement
        # so we pull the first index 
        high_weight_pos = self.masks.where(self.masks.WeightTooHigh)[0]
        if len(high_weight_pos) > 0:
            # after branching, we check if there is anything still
            # above the max weight threshold
            max_thresh = self.max_weight_threshold
            high_weights = weights[high_weight_pos]
            # from the walkers that originally had weights above the threshold
            # take the ones that _still_ have weights above the threshold
            still_too_high = np.where(high_weights > max_thresh)[0]
            
            # self.log_print("high_wpos {} \n still_high {}", 
            #     high_weight_pos,
            #     still_too_high,
            #     verbosity=self.logger.LogLevel.STATUS
            # )

            if len(still_too_high) > 0:
                # figure out the energies & weights of this shit heads
                # so that we can print out info about them
                num_high = len(still_too_high)
                still_too_high_pos = high_weight_pos[still_too_high]
                still_too_high_energies = energies[-1][still_too_high_pos]
                still_too_high_weights = weights[still_too_high_pos]

                self.log_print(
                    '\n    '.join([
                        'High-Weight Walkers: {}',
                        "Threshold: {}",
                        'Energy: Min {} | Max {} | Mean {}',
                        'Weight: Min {} | Max {} | Mean {}'
                    ]),
                    num_high,
                    max_thresh,
                    np.min(still_too_high_energies), np.max(still_too_high_energies), np.average(still_too_high_energies),
                    np.min(still_too_high_weights), np.max(still_too_high_weights), np.average(still_too_high_weights),
                    verbosity=self.logger.LogLevel.STATUS
                )

                # branch num_high low-weight walkers
                # implicitly, this _should_ branch the high-weight walkers
                # unless one of these walkers has such a high weight that it protects
                # one of the other high weight walkers from being branched
                # if this happens...we don't care because Anne asserts this should be very
                # unlikely
                eliminated_walkers = np.argpartition(weights, num_high)[:num_high]
                while len(eliminated_walkers) > 0:
                    eliminated_walkers = self.chop_weights(eliminated_walkers, weights, parents, walkers, energies)

        self.log_print(
            '\n    '.join([
                'Done branching:',
                'Energy: Min {} | Max {} | Mean {}',
                'Weight: Min {} | Max {} | Mean {}'
            ]),
            np.min(energies[-1]), np.max(energies[-1]), np.average(energies[-1]),
            np.min(weights), np.max(weights), np.average(weights),
            verbosity=self.logger.LogLevel.STATUS
        )

        # bond_lengths = np.linalg.norm(walkers[:, 1] - walkers[:, 0], axis=1)
        # self.log_print('Bond Lengths: Min {} | Max {} | Mean {} | Vib. Avg {}',
        #                np.min(bond_lengths), np.max(bond_lengths),
        #                np.average(bond_lengths), np.average(bond_lengths, weights=weights)
        #                )
        # self.log_print('Energies for Bonds: Min Dist. {} | Max {}',
        #                energies[-1, np.argmin(bond_lengths)], energies[-1, np.argmax(bond_lengths)]
        #                )
        return energies

    def descendent_weight(self):
        """Calls into the walker descendent weighting if the timing is right

        :return:
        :rtype:
        """
        if self.counter.equilibrated:
            status = self.counter.descendent_weighting_status
            self.log_print("Descendent Weighting Status: {}", status, verbosity=self.logger.LogLevel.STATUS)
            if status == self.counter.DescendentWeightingStatus.Complete:
                self.log_print("Collecting descendent weights", verbosity=self.logger.LogLevel.STATUS)
                dw = self.walkers.descendent_weight() # not sure where I want to cache these...
                self.wavefunctions.append(dw)
                self.num_wavefunctions += 1
                self.logger.save_wavefunction(dw)
            elif status == self.counter.DescendentWeightingStatus.Beginning:
                self.log_print("Starting descendent weighting propagation", verbosity=self.logger.LogLevel.STATUS)
                self.walkers._setup_dw()
        else:
            self.log_print("Equilibration not complete", verbosity=self.logger.LogLevel.STATUS)

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
