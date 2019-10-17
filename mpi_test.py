from __future__ import print_function
import sys, os, numpy as np, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
    default = 3,
    type = int,
    dest = 'walkers_per_core'
)
parser.add_argument(
    "--steps_per_call",
    default = 5,
    type = int,
    dest = 'steps_per_call'
)
parser.add_argument(
    "--iterations",
    default = 5,
    type = int,
    dest = 'iterations'
)
parser.add_argument(
    "--no_print",
    default = False,
    type = bool,
    dest = 'no_print'
)
parser.add_argument(
    "--displacement_radius",
    default = .5,
    type = float,
    dest = 'displacement_radius'
)
params = parser.parse_args()

########################################################################################################
#
#
#                                      RUN TEST
#
#
########################################################################################################
from RynLib import *

#
# initialize walkers
#
testWalker = np.array([
    [0.9578400,0.0000000,0.0000000],
    [-0.2399535,0.9272970,0.0000000],
    [0.0000000,0.0000000,0.0000000]
])
testAtoms = [ "H", "H", "O" ]

#
# set up MPI
#
who_am_i, _24601 = giveMePI()
num_walkers_per_core = params.walkers_per_core
if who_am_i == 0:
    num_walkers = _24601 * num_walkers_per_core
else:
    num_walkers = num_walkers_per_core

if who_am_i == 0:
    print("Number of processors / walkers: {} / {}".format(_24601, num_walkers))

#
# randomly permute things
#
testWalkersss = np.array( [ testWalker ] * num_walkers )
testWalkersss += np.random.uniform(low=-params.displacement_radius, high=params.displacement_radius, size=testWalkersss.shape)
test_iterations = params.iterations
test_results = np.zeros((test_iterations,))
lets_get_going = time.time()
nsteps = params.steps_per_call
testWalkersss = np.ascontiguousarray(
    np.broadcast_to(testWalkersss, (nsteps,) + testWalkersss.shape)
) # this forces contiguous memory layout

#
# run tests
#
for ttt in range(test_iterations):
    t0 = time.time()
    test_result = rynaLovesDMCLots(testAtoms, testWalkersss)
    test_results[ttt] = time.time() - t0
gotta_go_fast = time.time() - lets_get_going

#
# tell me how you really feel
#
if who_am_i == 0:

    if not params.no_print:
        print(
            # "Fed in: {}".format(testWalkersss),
            "Fed in walker array with shape {}".format(testWalkersss.shape),
            sep="\n"
        )
        print(
            "Got back: {}".format(test_result),
            "  with shape {}".format(test_result.shape),
            sep="\n"
            )
    else:
        print(
            "Got back result with shape {}".format(test_result.shape),
            sep="\n"
        )
    print("Total time: {}s (over {} iterations)".format(gotta_go_fast, test_iterations))
    print("Average total: {}s Average time per walker: {}s".format(np.average(test_results), np.average(test_results)/num_walkers/nsteps))


# raise Exception("why god why")
#
# tell MPI it did good
#

noMorePI()