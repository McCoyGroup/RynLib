from __future__ import print_function
import sys, os, numpy as np, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from RynLib import *

########################################################################################################
#
#
#                                      PARSE PARAMETERS
#
#
########################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "--walkers_per_core",
    default = 2,
    type = int,
    dest = 'walkers_per_core'
)
parser.add_argument(
    "--steps_per_call",
    default = 10,
    type = int,
    dest = 'steps_per_call'
)
parser.add_argument(
    "--name",
    default = "mpi_dmc",
    type = str,
    dest = 'name'
)
parser.add_argument(
    "--equilibration",
    default = 1000,
    type = int,
    dest = 'equilibration'
)
parser.add_argument(
    "--total_time",
    default = 10000,
    type = int,
    dest = 'total_time'
)
parser.add_argument(
    "--descendent_weighting_delay",
    default = 500,
    type = int,
    dest = 'descendent_weighting_delay'
)
parser.add_argument(
    "--descendent_weighting_steps",
    default = 50,
    type = int,
    dest = 'descendent_weighting_steps'
)
parser.add_argument(
    "--delta_tau",
    default = 5.0,
    type = float,
    dest = 'delta_tau'
)
parser.add_argument(
    "--checkpoint_every",
    default = 100,
    type = int,
    dest = 'checkpoint_every'
)
params = parser.parse_args()

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
num_walkers_per_core = params.walkers_per_core
if who_am_i == 0:
    num_walkers = _24601 * num_walkers_per_core
else:
    # we're just a single core, after all
    # THIS IS IMPORTANT BECAUSE I USE THIS ASSUMPTION ON THE MPI SIDE BECAUSE I WAS TOO LAZY TO ADD A PARAMETER TO THE CALL
    num_walkers = num_walkers_per_core

#
# Actual run parameters
#
dwDelay = params.descendent_weighting_delay
nDw = params.descendent_weighting_steps
deltaT = params.delta_tau
alpha = 1.0 / (2.0 * deltaT)
equil = params.equilibration
ntimeSteps = params.total_time
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

thank_you_victor = "water_start.dat"

#
#   Initialize walker set
#
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
        walkers_new = Constants.convert(walkers, "angstroms", in_AU=False) #np.ndarray
        walkers_new = np.ascontiguousarray(walkers_new.transpose(1, 0, 2, 3))
        # this makes our walkers the right shape to be used at the C++ level
        # this is actually low-key _crucial_ to the code working correctly
        res = rynaLovesDMCLots(atoms, walkers_new)
        res = res.transpose()
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
    params.name,
    """ A simple water scan using Entos and MPI to get the potential (and including ML later)""",
    walker_set = walkers,

    time_step = deltaT,
    D = 1/2.0,
    alpha = alpha,

    potential = potential,

    num_time_steps = ntimeSteps,
    steps_per_propagation = params.steps_per_call,
    equilibration = equil,

    checkpoint_at = params.checkpoint_every,

    descendent_weighting = (dwDelay, nDw),
    write_wavefunctions = True,

    world_rank = who_am_i,
    verbosity=Simulation.LOG_DEBUG

)

#
#   Run simulation
#

out = sim.log_file

if not os.path.exists(os.path.dirname(out)):
    try:
        os.makedirs(os.path.dirname(out))
    except OSError:
        pass
try:
    with open(sim.log_file, 'w+'):
        pass
except OSError:
    pass

if who_am_i == 0:
    start = time.time()
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
    sim.checkpoint(test=False)
else:
    sim.log_print("    core {} finished".format(who_am_i), verbosity=7)