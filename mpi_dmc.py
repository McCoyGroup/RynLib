from __future__ import print_function
import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
        if res is not None:
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