import os, sys

lib_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, lib_dir)
try:
    from .src.Dumpi import *
except ImportError:
    from .src.setup import failed
    if failed:
        raise
    from .src.Dumpi import *
finally:
    sys.path.remove(lib_dir)