import numpy as np, math, numpy.linalg as la
import sys, os, copy, time
import subprocess as sp

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
        return val * vv if in_AU else val / vv

    @classmethod
    def mass(cls, atom, in_AU = True):
        m = cls.masses[atom]
        if in_AU:
            m = cls.convert(m, "amu")
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
                 walker_set = None, descendent_weighting_delay = None, num_time_steps = None,
                 time_step = None, D = None, alpha = None,
                 potential = None, log_file = sys.stdout
                 ):
        from collections import deque

        self.walkers = walker_set
        self.energies = deque() # just a convenient data structure to push into
        self.descendent_weighting_delay = descendent_weighting_delay
        self._last_desc_weighting_step = 0
        self.stack = deque(self.descendent_weighting_delay)
        self.step_num = 0
        self.num_time_steps = num_time_steps
        self.time_step = time_step
        self.D = D # not sure what this is supposed to be...?
        self.alpha = alpha
        self.potential = potential
        self.log_file = log_file

    def log_print(self, *message, **kwargs):
        if not self.log_file is None:
            log = self.log_file
            if isinstance(log, str):
                with open(log, "a") as lf: # this is potentially quite slow but I am also quite lazy
                    print(*message, file = lf, **kwargs)
            else:
                print(*message, file = log, **kwargs)

    folders = ['wts']
    def _init_storage(self):
        for fold in self.folders:
            if not os.path.exists(fold):
                os.makedirs(fold)

    @property
    def _do_descendent_weighting(self):
        return self.step_num - self._last_desc_weighting_step >= self.descendent_weighting_delay

    def propagate(self, nsteps = 1):
        coord_sets = self.walkers.get_displaced_coords(nsteps)
        energies = self.potential(self.walkers.atoms, coord_sets)
        self.step_num += nsteps
        new_weights, new_walkers = self.branch(coord_sets, energies)
        self.walkers.weights = new_weights
        self.walkers.coords = new_walkers
        if self.dodw:
            self._last_desc_weighting_step = self.step_num
        return energies

    def compute_vref(self, energies):
        # we'll assume multiple time steps of energies
        vrefs = np.average(energies, axis=1)
        Vbar = np.average(varray,weights=cont_wt)
        correction=(np.sum(cont_wt-np.ones(initialWalkers)))/initialWalkers
        vref = Vbar - (alpha * correction)
        return vrefs

    def branch(self, walkers, energies):
        weights = self.walkers.weights
        for w, e, v in
    def _branch_1(self, weights, walkers, energies, vrefs):
        # we'll assume multiple walker steps and multiple energies -- we can special case the singular version later


        weights *= np.product(np.exp(-1.0 * (energies - vrefs) * self.time_step))
        # we want to consider the cumulative weight of *all* the steps I think...?

        threshold = 1.0 / self.walkers.num_walkers
        eliminated_walkers = np.argwhere(cont_wt < threshold)
        self.log_print('Walkers being removed: {}'.format(len(eliminated_walkers)))
        self.log_print('Max weight in ensemble: {}'.format(np.amax(weights)))

        walkers = walkers[-1]
        for dying in eliminated_walkers: # gotta do it iteratively to get the max_weight_walker right..
            cloning = np.argmax(weights)
            walkers[dying] = walkers[cloning]
            weights[dying] = weights[cloning] / 2.0
            weights[cloning] /= 2.0

        return walkers, weights

    def descendent_weight(self):
        pass

class WalkerSet:
    def __init__(self, atoms = None, initial_walker = None, num_walkers = None):
        self.n = len(atoms)
        self.atoms = atoms
        self.masses = np.array([ Constants.mass(a) for a in self.atoms ])
        self.coords = np.array([ initial_walker ]) * num_walkers
        self.parents = np.arange(num_walkers)
        self.num_walkers = num_walkers
        self.weights = np.ones(num_walkers)
    def initialize(self, deltaT, D):
        self.deltaT = deltaT
        self.sigmas = np.sqrt((2 * D * deltaT) / self.masses)
    def get_displacements(self, steps = 1):
        return np.array([
            np.random.normal(0.0, sig, (steps, len(self.coords), 3)) for sig in self.sigmas
        ]).T
    def get_displaced_coords(self, n=1):
        accum_disp = np.cumsum(self.get_displacements(n))
        return self.coords + accum_disp # hoping the broadcasting makes this work...
    def snapshot(self, file):
        import pickle
        pickle.dump(self, file) # this could easily be replaced with numpy.savetxt though

class Plotter:
    _mpl_loaded = False
    @classmethod
    def load_mpl(cls):
        if not cls._mpl_loaded:
            import matplotlib as mpl
            mpl.use('Agg')
            cls._mpl_loaded = True
    @classmethod
    def plot_e_ref(cls, e_refs):
        import matplotlib.pyplot as plt
        pass


if __name__ == "__main__":

    walkers = WalkerSet(
        atoms = Constants.water_structure[0],
        initial_walker = Constants.water_structure[1],
        num_walkers = 3000
    )

    nDw = 30
    deltaT = 5.0
    alpha = 1.0 / (2.0 * deltaT)
    ntimeSteps = 10000 + nDw
    sim = Simulation(
        descendent_weighting_delay = nDw,
        num_time_steps = ntimeSteps,
        time_step = 5.0,
        alpha = alpha
    )
    
def moveRandomly():
    # choose a random number from gaussian distribution (1/2pisig)(e^(-dx^2/1sig^2))
    for p in range(len(myWalkers)):
        for atom in range(len(myWalkers[p].coords)):
            gaussStep = np.random.normal(loc=0.00000000000000, scale=sigma[atom], size=3)
            myWalkers[p].coords[atom] += gaussStep

# Start!

######Initialize Simulation Data Structures############
nSim=0
nBranchSteps=10
myWalkers = [Walker() for r in range(initialWalkers)]
cont_wt = np.ones(initialWalkers)
start = time.time()
vrefAr = np.zeros(ntimeSteps)
#Initialize Potential

getPotentialForWalkers()
Vref = getVref()
reunion = False
DW = False
firstWtar = np.zeros(ntimeSteps)
logFile = open('log.txt','w+')
weightingCycle = np.arange(500,ntimeSteps,500)[1:]+1 #every 500 t.s. we are collecting descendent weights for nDw timesteps
endWeightingCycle = weightingCycle + nDw
logFile.write('weightingCycle '+str(weightingCycle)+'\n')
logFile.write('endWeightingCycle '+str(endWeightingCycle)+'\n')

branchCycle = np.arange(nBranchSteps,ntimeSteps,nBranchSteps)

#print 'weightingCycle',weightingCycle
#print 'endWeightingCycle',endWeightingCycle 
#######################################################
for i in range(ntimeSteps):
    print i
    logFile.write("time step "+str(i)+'\n')
    if i in weightingCycle:
        #print 'weighting begins',i
        xyzExportCoords([walkers.coords for walkers in myWalkers],'h2oCoords_wfn_'+str(nSim))
        DW = True
        logFile.write('commence descendent weighting\n')
        pointer = np.arange(initialWalkers)
        #print 'saving contwt',cont_wt
        np.savetxt('wts/wtsBeforeDw_numWalkers_'+str(initialWalkers)+'wfn'+str(nSim),cont_wt)
    moveRandomly()
    if i == 0:
        Vref = getVref()
    if i in branchCycle:
        birthOrDeath(DW)
    else:
        getMrkPotentialForWalkers()
        for y in range(len(myWalkers)):
            curV = myWalkers[y].WalkerV
            cont_wt[y] *= np.exp(-1.0 * (curV - Vref) * deltaT)
    Vref = getVref()
    if i in (endWeightingCycle-1):
        DW = False
        #print 'weighting ends',i
        #sum up descendent weights - if there was branching from cont. weighting take that into account. 
        dweights=np.zeros(initialWalkers)
        for q in range(len(cont_wt)):
            dweights[q]= np.sum(cont_wt[pointer==q])
        #print 'dweights over, dweits=',dweights
        np.savetxt('wts/wtsAfterDw_numWalkers_'+str(initialWalkers)+'wfn'+str(nSim), dweights)
        np.savetxt('wts/pointer_numWalkers_'+str(initialWalkers)+'wfn'+str(nSim), pointer)
        np.savetxt('wts/justwtsAfter'+str(initialWalkers)+'wfn'+str(nSim),cont_wt)
        nSim+=1
    # Plotting business
    vrefAr[i] = Vref
    firstWtar[i]=cont_wt[0]
logFile.write('that took ' + str(time.time() - start) + ' seconds' + '\n')
np.savetxt('VrefArray.txt', vrefAr/wvnmbr)
np.savetxt('FirstWalkersWeightOverTheSimulation.txt',firstWtar)
logFile.close()
#END!