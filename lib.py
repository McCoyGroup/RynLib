import os, sys

lib_dir = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.insert(0, lib_dir)
    from .RynLib import *
except ImportError:
    from .setup import failed
    if failed:
        raise
    from .RynLib import *
finally:
    sys.path.pop(0)