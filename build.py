import sys, os
mine_mine_mine = os.path.dirname(os.path.abspath(__file__))
try:
    os.remove(os.path.join(mine_mine_mine, "RynLib.so"))
except:
    pass
sys.path.insert(0, mine_mine_mine)

import test
