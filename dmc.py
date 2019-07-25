from __future__ import print_function
import numpy as np, os, time

##############################################################################################################
#
#                                       Simulation Classes
#
class Constants:
    atomic_units = {
        "wavenumbers" : 4.55634e-6,
        "angstroms" : 0.529177,
        "amu" : 1.000000000000000000/6.02213670000e23/9.10938970000e-28   #1822.88839  g/mol -> a.u.
    }

    masses = {
        "H" : ( 1.00782503223, "amu"),
        "O" : (15.99491561957, "amu")
    }

    @classmethod
    def convert(cls, val, unit, in_AU = True):
        vv = cls.atomic_units[unit]
        return (val * vv) if in_AU else (val / vv)

    @classmethod
    def mass(cls, atom, in_AU = True):
        m = cls.masses[atom]
        if in_AU:
            m = cls.convert(*m)
        return m

    water_structure = (
        ["H", "H", "O" ],
        np.array(
            [[0.9578400,0.0000000,0.0000000],
             [-0.2399535,0.9272970,0.0000000],
             [0.0000000,0.0000000,0.0000000]]
        )
    )

class Simulation:
    def __init__(self,
                 name = "dmc",
                 description = "a dmc",
                 walker_set = None,
                 D = None, time_step = None,
                 steps_per_propagation = None,
                 num_time_steps = None,
                 alpha = None,
                 potential = None,
                 equilibration = None,
                 descendent_weighting = None,
                 write_wavefunctions = None,
                 checkpoint_at = None,
                 zpe_averages = 1000,
                 output_folder = None,
                 log_file = None,
                 verbosity = 0,
                 dummied = None, # for MPI usage
                 world_rank = 0
                 ):
        """Sets up all the necessary simulation data to run a DMC

        :param name: name to be used when storing file data
        :type name: str
        :param description: long description which isn't used for anything
        :type description: str
        :param walker_set: the WalkerSet object that handles all the pure walker activities in the simulation
        :type walker_set: WalkerSet
        :param D: ??? usually it's 2.0 ???
        :type D: float
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

        # store params for snapshotting
        self._params = dict(
            name= name,
            description = description,
            D = D,
            time_step = time_step,
            steps_per_propagation = steps_per_propagation,
            num_time_steps = num_time_steps,
            alpha = alpha,
            equilibration = equilibration,
            descendent_weighting = descendent_weighting,
            write_wavefunctions = write_wavefunctions,
            checkpoint_at = checkpoint_at,
            zpe_averages = zpe_averages,
            output_folder = output_folder,
            log_file = log_file,
            verbosity = verbosity,
            dummied = dummied, # for MPI usage
            world_rank = world_rank
        )

        from collections import deque

        self.name = name
        self.description = description

        self.walkers = walker_set
        self.potential = potential

        self.alpha = alpha
        self.reference_potentials = deque() # just a convenient data structure to push into

        self.step_num = 0
        self.num_time_steps = num_time_steps
        self.prop_steps = steps_per_propagation

        self.time_step = time_step
        self.D = D # not sure what this is supposed to be...?
        self.walkers.initialize(self, time_step, D)

        self._dw_delay, self._dw_steps = descendent_weighting
        self._write_wavefunctions = write_wavefunctions
        if write_wavefunctions:
            self.wavefunctions = deque(maxlen=1)
        else:
            self.wavefunctions = deque()
        self._num_wavefunctions = 0 # here so we can do things with save_wavefunction
        self._last_dw_step = 0
        self._dw_initialized_step = None

        if output_folder is None:
            output_folder = os.path.join(os.path.abspath("dmc_data"), self.name)
        self.output_folder = output_folder

        if log_file is None:
            log_file = os.path.join(self.output_folder, "log.txt")
        self.log_file = log_file

        self.verbosity = verbosity

        self._equilibrated = False
        if isinstance(equilibration, int):
            equilibration = lambda s, e=equilibration: s.step_num > e #classic lambda parameter binding
        self.equilibration_check = equilibration

        if dummied is None:
            dummied = world_rank != 0
        self.dummied = dummied
        self.world_rank = world_rank

        self._previous_checkpoint = 0
        if isinstance(checkpoint_at, int):
            checkpoint_at = lambda s, c=checkpoint_at: s.step_num - s._previous_checkpoint > c
        elif checkpoint_at is None:
            checkpoint_at = lambda *a: False
        self._checkpoint = checkpoint_at

        self.zpe_averages = zpe_averages

    LOG_BASIC = 1
    LOG_STATUS = 3
    LOG_STEPS = 5
    LOG_MPI = 7
    LOG_ALL = 10
    LOG_DEBUG = 100
    def log_print(self, message, *params, **kwargs):
        if 'verbosity' in kwargs:
            verbosity = kwargs['verbosity']
            del kwargs['verbosity']
        else:
            verbosity = self.LOG_BASIC
        if verbosity <= self.verbosity and self.log_file is not None:
            log = self.log_file
            if isinstance(log, str):
                if not os.path.isdir(os.path.dirname(log)):
                    os.makedirs(os.path.dirname(log))
                #O_NONBLOCK is *nix only
                with open(log, "a", os.O_NONBLOCK) as lf: # this is potentially quite slow but I am also quite lazy
                    print(message.format(*params), file = lf, **kwargs)
            else:
                print(message.format(*params), file = log, **kwargs)

    @property
    def zpe(self):
        return self.get_zpe()
    def get_zpe(self, n = None):
        import itertools
        if n is None:
            n = self.zpe_averages
        if len(self.reference_potentials) > n:
            vrefs = np.array(itertools.islice(self.reference_potentials, len(self.reference_potentials)-n, 1))
        else:
            vrefs = self.reference_potentials
        return np.average(np.array(vrefs))

    @property
    def equilibrated(self):
        if not self._equilibrated:
            self._equilibrated = self.equilibration_check(self)
        return self._equilibrated

    def checkpoint(self):
        if self._checkpoint(self):
            self.log_print("Checkpointing simulation", verbosity=self.LOG_STEPS)
            # self.snapshot("checkpoint.pickle")
            self.snapshot_energies()
            self.snapshot_params()
            self.snapshot_walkers()

    @classmethod
    def reload(cls,
               output_folder,
               params_file = "params.pickle",
               energies_file = 'energies.npy'
               ):
        """Reloads a Simulation object from a director with specified params file

        :param core_dir:
        :type core_dir:
        :param params_file:
        :type params_file:
        """
        import pickle

        params_file = os.path.join(output_folder, params_file) if not os.path.isfile(params_file) else params_file
        with open(params_file) as pf:
            params = pickle.load(pf)

        energies_file = os.path.join(output_folder, energies_file) if not os.path.isfile(energies_file) else energies_file
        energies = np.load(energies_file)


    def snapshot(self, file="snapshot.pickle"):
        """Saves a snapshot of the simulation to file

        :param file:
        :type file:
        :return:
        :rtype:
        """
        raise NotImplementedError("Turns out pickle doesn't like this")

        import pickle

        f = os.path.abspath(file)
        if not os.path.isfile(f):
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder)
            f = os.path.join(self.output_folder, file)
        with open(f, "w+") as binary:
            pickle.dump(self, binary)

    def snapshot_params(self, file="params.pickle"):
        """Saves a snapshot of the params to a pickle

        :return:
        :rtype:
        """
        import pickle

        self._params.update(step_num = self.step_num)
        f = os.path.abspath(file)
        if not os.path.isfile(f):
            f = os.path.join(self.output_folder, file)
        out_dir = os.path.dirname(f)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        with open(f, "w+") as binary:
            pickle.dump(self._params, binary)
    def snapshot_walkers(self, file="walkers.npz"):
        """Saves a snapshot of the walkers to a pickle

        :return:
        :rtype:
        """
        f = os.path.abspath(file)
        if not os.path.isfile(f):
            f = os.path.join(self.output_folder, file)
        out_dir = os.path.dirname(f)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.walkers.snapshot(file)
    def save_wavefunction(self):
        """Save wavefunctions to a numpy binary

        :return:
        :rtype:
        """
        wf_dir = os.path.join(self.output_folder, "wavefunctions")
        if not os.path.isdir(wf_dir):
            os.makedirs(wf_dir)
        file = os.path.join(wf_dir, 'wavefunction_{}.npz'.format(self._num_wavefunctions))
        if self.verbosity:
            self.log_print("Saving wavefunction to {}", file, verbosity=self.LOG_STEPS)
        np.savez(file, **self.wavefunctions[-1])
        return file
    def snapshot_energies(self, file="energies"):
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
        np.save(f, np.array(self.reference_potentials))
        return f

    def run(self):
        """Runs the DMC until we've gone through the requested number of time steps

        :return:
        :rtype:
        """
        while self.step_num < self.num_time_steps:
            self.propagate()

    def propagate(self, nsteps = None):
        """Propagates the system forward n steps

        :param nsteps: number of steps to propagate for; None means automatic
        :type nsteps:
        :return:
        :rtype:
        """
        if nsteps is None:
            nsteps = self.prop_steps

        if not self.dummied:
            self.log_print("Starting step {}", self.step_num, verbosity=self.LOG_STATUS)
            self.log_print("Moving coordinates {} steps", nsteps, verbosity=self.LOG_STEPS)
            coord_sets = self.walkers.displace(nsteps)
            self.log_print("Computing potential energy", verbosity=self.LOG_STATUS)
            start = time.time()
            energies = self.potential(self.walkers.atoms, coord_sets, sim=self)
            end = time.time()
            self.log_print("    took {}s", end-start, verbosity=self.LOG_STATUS)
            self.step_num += nsteps
            self.log_print("Updating walker weights", verbosity=self.LOG_STEPS)
            weights = self.update_weights(energies, self.walkers.weights)
            self.walkers.weights = weights
            self.log_print("Branching", verbosity=self.LOG_STEPS)
            self.branch()
            self.log_print("Applying descendent weighting", verbosity=self.LOG_STEPS)
            self.descendent_weight()
            self.checkpoint()
            if self.verbosity >= self.LOG_STATUS: # we do the check here so as to not waste time computing ZPE... even though that waste is effectively 0
                self.log_print("Zero-point Energy: {}", self.zpe, verbosity=self.LOG_STATUS)
        else:
            # self.log_print("Starting step {} on core {}", self.step_num, verbosity=self.LOG_STEPS)
            self.log_print("    computing potential energy on core {}", self.world_rank, verbosity=self.LOG_MPI)
            try:
                walk = self._dummy_walkers
            except AttributeError:
                self._dummy_walkers = np.broadcast_to(self.walkers.coords, (nsteps,) + self.walkers.coords.shape).copy()
                walk = self._dummy_walkers
            self.potential(self.walkers.atoms, walk, sim=self)
            self.step_num += nsteps
        # Plotter.plot_psi(self)

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
            self.log_print("Energies: {}", e, verbosity=self.LOG_DEBUG)
            Vref = self._compute_vref(e, weights)
            self.reference_potentials.append(Vref) # a constant time operation
            new_wts = np.nan_to_num(np.exp(-1.0 * (e - Vref) * self.time_step))
            weights *= new_wts
            self.log_print("Weights: {}", weights, verbosity=self.LOG_DEBUG)
        return weights

    def branch(self):
        return self.walkers.branch()

    def descendent_weight(self):
        """Calls into the walker descendent weighting if the timing is right

        :return:
        :rtype:
        """
        if self.equilibrated:
            step = self.step_num
            if (self._dw_initialized_step is not None) and step - self._dw_initialized_step >= self._dw_steps:
                self.log_print("Collecting descendent weights at time step {}", step, verbosity=self.LOG_STATUS)
                dw = self.walkers.descendent_weight() # not sure where I want to cache these...
                self.wavefunctions.append(dw)
                self._num_wavefunctions += 1
                if self._write_wavefunctions:
                    self.save_wavefunction()
                self._dw_initialized_step = None
            elif step - self._last_dw_step >= self._dw_delay:
                if self.verbosity:
                    self.log_print("Starting descendent weighting propagation at time step {}", step, verbosity=self.LOG_STATUS)
                self.walkers._setup_dw()
                self._dw_initialized_step = step
                self._last_dw_step = step

class WalkerSet:
    def __init__(self, atoms = None, masses = None, initial_walker = None, num_walkers = None):
        self.n = len(atoms)
        self.num_walkers = num_walkers

        self.atoms = atoms
        if masses is None:
            masses = np.array([ Constants.mass(a) for a in self.atoms ])
        self.masses = masses
        self.coords = np.array([ initial_walker ] * num_walkers)
        self.weights = np.ones(num_walkers)

        self.parents = np.arange(num_walkers)
        self.sigmas = None
        self._parents = self.coords.copy()
        self._parent_weights = self.weights.copy()

    def initialize(self, sim, deltaT, D):
        """Sets up necessary parameters for use in calculating displacements and stuff

        :param deltaT:
        :type deltaT:
        :param D:
        :type D:
        :return:
        :rtype:
        """
        self.deltaT = deltaT
        if self.sigmas is not None:
            self.sigmas = np.sqrt((2 * D * deltaT) / self.masses)
        self.log_print = sim.log_print # this makes it abundantly clear that branching should *not* be on the WalkerSet
    def get_displacements(self, steps = 1):
        shape = (steps, ) + self.coords.shape[:-2] + self.coords.shape[-1:]
        disps = np.array([
            np.random.normal(0.0, sig, size = shape) for sig in self.sigmas
        ])

        disps = np.transpose(disps, (1,2,0,3))

        return disps
    def get_displaced_coords(self, n=1):
        # accum_disp = np.cumsum(self.get_displacements(n), axis=1)
        # return np.broadcast_to(self.coords, (n,) + self.coords.shape) + accum_disp # hoping the broadcasting makes this work...

        crds = np.zeros((n,) + self.coords.shape, dtype='float')
        bloop = self.coords
        disps = self.get_displacements(n)
        for i, d in enumerate(disps): # loop over atoms
            bloop = bloop + d
            crds[i] = bloop

        return crds

    def displace(self, n=1):
        coords = self.get_displaced_coords(n)
        self.coords = coords[-1]
        return coords
    def branch(self):
        """Handles branching in the system.

        :return:
        :rtype:
        """

        weights = self.weights
        walkers = self.coords
        parents = self.parents
        threshold = 1.0 / self.num_walkers
        eliminated_walkers = np.argwhere(weights < threshold).flatten()
        self.log_print('Walkers being removed: {}', len(eliminated_walkers), verbosity=Simulation.LOG_STATUS) # -__-
        self.log_print('Max weight in ensemble: {}', np.amax(weights), verbosity=Simulation.LOG_STATUS)

        for dying in eliminated_walkers: # gotta do it iteratively to get the max_weight_walker right..
            cloning = np.argmax(weights)
            # print(cloning)
            parents[dying] = parents[cloning]
            walkers[dying] = walkers[cloning]
            weights[dying] = weights[cloning] / 2.0
            weights[cloning] /= 2.0

    def _setup_dw(self):
        self.parents = np.arange(self.num_walkers)
        self._parents = self.coords.copy()
        self._parent_weights = self.weights.copy()
    def descendent_weight(self):
        """Handles the descendent weighting in the system

        :return: tuple of parent coordinates, descendend weights, and original weights
        :rtype:
        """

        weights = np.array( [ np.sum(self.weights[ self.parents == i ]) for i in range(self.num_walkers) ] )
        descendent_weights = {"coords":self._parents, "weights":weights, "original_weights":self._parent_weights}

        return descendent_weights

    def snapshot(self, file):
        """Snapshots the current walker set to file"""

        np.savez(file,
                 coords=self.coords, weights=self.weights, sigmas=self.sigmas,
                 parents=self.parents,
                 parent_coords=self._parents,
                 parent_weights = self._parent_weights
                 )

    @classmethod
    def load(cls, file, atoms = None, masses = None):
        """Reloads WalkerSet from file"""

        npz = np.load(file)

        sigs = npz['sigmas']

        if masses is None:
            masses = np.ones(len(sigs))
        if atoms is None:
            atoms = ['H'] * len(sigs)

        coords = npz['coords']
        self = cls(atoms = atoms, masses = masses, initial_walker=coords, num_walkers=len(coords))

        self.parents = npz['parents']
        self.sigmas = npz['sigmas']
        self.weights = npz['weights']
        self.parents = npz['parents']
        self._parent_weights = npz['parent_weights']
        self._parent_coords = npz['parent_coords']

        return self

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
        import matplotlib.pyplot as plt
        e = np.array(sim.reference_potentials)
        n = np.arange(len(e))
        fig, axes = plt.subplots()
        e=Constants.convert(e,'wavenumbers',in_AU=False)
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

        hist, bins = np.histogram(w.coords.flatten(), weights=(w.weights), bins = 20, density = True)
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

        hist, bins = np.histogram(coord, weights=dw, bins = 20, density = True)
        bins -= (bins[1] - bins[0]) / 2
        axes.plot(bins[:-1], hist)
        plt.show()


if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from RynLib.loader import *

    walkers = WalkerSet(
        atoms = Constants.water_structure[0],
        initial_walker = Constants.water_structure[1] * 1.01, # inflate it a bit
        num_walkers = 100
    )

    dwDelay = 500
    nDw = 30
    deltaT = 5.0
    alpha = 1.0 / (2.0 * deltaT)
    equil = 1000
    ntimeSteps = 10000
    ntimeSteps += nDw

    # def exportCoords(cds, fn):  # for partridge schwinke
    #     cds = np.array(cds)
    #     fl = open(fn, "w+")
    #     fl.write('%d\n' % len(cds))
    #     for i in range(len(cds)):  # for a walker
    #         for j in range(len(cds[i])):  # for a certain # of atoms
    #             fl.write('%5.16f %5.16f %5.16f\n' % (cds[i, j, 0], cds[i, j, 1], cds[i, j, 2]))
    #     fl.close()
    # def PatrickShinglePotential(atms,walkerSet):
    #     import subprocess as sub
    #     bigPotz = np.zeros(walkerSet.shape[:2])
    #     for i,coordz in enumerate(walkerSet):
    #         exportCoords(coordz, 'PES/PES0' + '/hoh_coord.dat')
    #         proc = sub.Popen('./calc_h2o_pot', cwd='PES/PES0')
    #         proc.wait()
    #         bigPotz[i] = np.loadtxt('PES/PES0' + '/hoh_pot.dat')
    #     return bigPotz


    sim = Simulation(
        "water_simple",
        """ A simple water scan using Entos and MPI to get the potential (and including ML later)""",
        walker_set = walkers,
        time_step = 5.0, D = 1/2.0,

        alpha = alpha,
        potential = rynaLovesDMCLots,

        num_time_steps = ntimeSteps,
        steps_per_propagation = 1,
        equilibration = equil,

        descendent_weighting = (dwDelay, nDw),
        output_folder = os.path.expanduser("~/Desktop"),

        write_wavefunctions = False,
        log_file=sys.stdout

    )

    # def hoop(atoms, rs, w = 1):
    #     pots = w * np.power(rs, 2)
    #     pots = pots.reshape(pots.shape[:2])
    #     return pots
    # ho_walkers = WalkerSet(
    #     atoms = [ "H" ],
    #     initial_walker = np.array([ [ .01 ] ]),
    #     num_walkers = 10000
    # )
    # sim = Simulation(
    #     "ho",
    #     """Test harmonic oscillator""",
    #     walker_set = ho_walkers,
    #     time_step = 5.0, D = 1/2.0,
    #
    #     alpha = alpha,
    #     potential = hoop,
    #
    #     num_time_steps = ntimeSteps,
    #     steps_per_propagation = 10,
    #
    #     descendent_weighting = (dwDelay, nDw),
    #     equilibration = equil,
    #     write_wavefunctions = False,
    #
    #     output_folder = os.path.expanduser("~/Desktop"),
    #     log_file = sys.stdout,
    #     verbosity=0
    # )

    try:
        sim.run()
    except ZeroDivisionError:
        import traceback as tb
        tb.print_exc()

    Plotter.plot_vref(sim)
    # Plotter.plot_psi(sim)
    # Plotter.plot_psi2(sim)