from __future__ import print_function
import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from RynLib import *

testWalker = np.array([
    [0.9578400,0.0000000,0.0000000],
    [-0.2399535,0.9272970,0.0000000],
    [0.0000000,0.0000000,0.0000000]
])
testAtoms = [ "H", "H", "O" ]
who_am_i, _24601 = giveMePI()
if who_am_i == 0:
    print("Number of processors (and walkers): {}".format(_24601))
testWalkersss =np.array( [ testWalker ] * _24601 )
testWalkersss += np.random.uniform(low=-.5, high=.5, size=testWalkersss.shape)
test_iterations = 50
test_results = np.zeros((test_iterations,))
lets_get_going = time.time()
for ttt in range(test_iterations):
    t0 = time.time()
    test_result = rynaLovesDMCLots(testAtoms, testWalkersss)
    test_results[ttt] = time.time() - t0
gotta_go_fast = time.time() - lets_get_going
if who_am_i == 0:
    print("Got back: {}".format(test_result))
    print("Total time: {}s (over {} iterations)".format(gotta_go_fast, test_iterations))
    print("Average total: {}s Average time per walker: {}s".format(np.average(test_results), np.average(test_results)/_24601))
noMorePI()
