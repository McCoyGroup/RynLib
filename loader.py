import os, sys

lib_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(lib_dir, "lib"))
sys.path.insert(0, lib_dir)
try:
    from .RynLib import *
except ImportError:
    from .src.setup import failed
    if failed:
        raise
    from .RynLib import *
finally:
    sys.path.pop(0)