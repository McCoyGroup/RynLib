import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DoMyCode import *
from AuntPetuniasRunnerBeans import *

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