import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from RynLib import *

import numpy as np
testWalker = np.array([
    [0.9578400,0.0000000,0.0000000],
    [-0.2399535,0.9272970,0.0000000],
    [0.0000000,0.0000000,0.0000000]
])
testAtoms = [ "H", "H", "O" ]
print(rynaLovesDMC(testAtoms, testWalker))