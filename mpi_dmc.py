from __future__ import print_function
import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from RynLib import *

########################################################################################################
#
#
#                                      PARAMETERS WOWWWWWWWWWW
#
#
########################################################################################################

#
# Number of walkers
#
# Sets up the MPI part of the thing. We use _24601 processors and thus _24601 * num_walkers_per_core walkers.
who_am_i, _24601 = giveMePI()
num_walkers_per_core = 3
if who_am_i == 0:
    num_walkers = _24601 * num_walkers_per_core
else:
    # we're just a single core, after all
    num_walkers = num_walkers_per_core

#
# Actual run parameters
#
dwDelay = 500
nDw = 50
deltaT = 5.0
alpha = 1.0 / (2.0 * deltaT)
equil = 1000
ntimeSteps = 10000
ntimeSteps += nDw

########################################################################################################
#
#
#                                     SIMULATE FOR REAL NOW GUYS
#
#
########################################################################################################
if who_am_i == 0:
    print("Number of processors / walkers: {} / {}".format(_24601, num_walkers))

# once upon a time a long long time ago I was <s>a ho</s> working with structures Victor gave me -- no more
def load_walkers(init_file, how_many_fren_u_hav = num_walkers):
    """Loads walkers configurations from an XYZ laid out like:
    OX OY OZ H1X H1Y H1Z H2X H2Y H2Z

    :param init_file: XYZ file to load from
    :type init_file:
    :param how_many_fren_u_hav: how many walkers are in the simulation
    :type how_many_fren_u_hav:
    """

    help_me_johnny = np.loadtxt(init_file)
    how_many_fren_i_hav = len(help_me_johnny)

    if how_many_fren_u_hav < how_many_fren_i_hav:
        help_me_johnny = help_me_johnny[:how_many_fren_u_hav]
    elif how_many_fren_u_hav > how_many_fren_i_hav:
        down_by_5 = how_many_fren_u_hav - how_many_fren_i_hav
        replicons = 1 + (down_by_5 // how_many_fren_i_hav)
        help_me_johnny = np.stack([help_me_johnny]*replicons)[:how_many_fren_u_hav]

    help_me_johnny = np.reshape(help_me_johnny, (how_many_fren_u_hav, 3, 3))
    return help_me_johnny


#
#   Initialize walker set
#
thank_you_victor = "water_start.dat"
walkers = WalkerSet(
    atoms = Constants.water_structure[0],
    initial_walker = Constants.convert(Constants.water_structure[1] * 1.01, "angstroms", in_AU=True), # inflate it a bit ##load_walkers(thank_you_victor)
    num_walkers = num_walkers
)


#
#   Define potential over our walkers
#
def potential(atoms, walkers, sim=None):
    import io
    fake_stderr = io.StringIO()
    try:
        real_stderr = sys.stderr
        sys.stderr = fake_stderr
        res = rynaLovesDMCLots(atoms, Constants.convert(walkers, "angstroms", in_AU=False))
    finally:
        sys.stderr = real_stderr

    err_msg = fake_stderr.getvalue()
    if err_msg:
        sim.log_print('\ton potential call got error\n\t\t{}\n\tfor walkers\n\t\t{}', err_msg, walkers)

    holdMyPI()
    return res


#
#   Initialize simulation
#
sim = Simulation(
    "water_simple",
    """ A simple water scan using Entos and MPI to get the potential (and including ML later)""",
    walker_set = walkers,
    time_step = 5.0, D = 1/2.0,

    alpha = alpha,
    potential = potential,

    num_time_steps = ntimeSteps,
    steps_per_propagation = 10,
    equilibration = equil,

    checkpoint_at = 100,

    descendent_weighting = (dwDelay, nDw),
    write_wavefunctions = True,

    world_rank = who_am_i,
    verbosity=Simulation.LOG_DEBUG

)


# Can't remember what this is for???
out = sim.log_file
line_buffering = 1

#
#   Run simulation
#
if who_am_i == 0:
    start = time.time()
    with open(sim.log_file, 'w+'):
        pass
    sim.log_print("Starting DMC simulation")
else:
    sim.log_print("    starting simulation on core {}".format(who_am_i), verbosity=7)

holdMyPI() # blocks until all our friends arrive
sim.run()

#
#   Tell me I did good, Pops ;_;
#
if who_am_i == 0:
    end = time.time()
    elapsed = end-start
    sim.log_print("Simulation finished in {}s.\nZPE = {}".format(elapsed, sim.zpe))
    sim.snapshot()
else:
    sim.log_print("    core {} finished".format(who_am_i), verbosity=7)