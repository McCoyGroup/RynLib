"""
Top-level build file that will precompile the various portions of the project that should be precompiled in a
containerized environment
"""
import os

os.chdir(os.path.dirname(__file__))

from RynLib.Interface import RynLib

print("--INSTALLING OPEN MPI--")
RynLib.install_MPI()

print("--MOUNTING VOLUMES FOR DATA READ/WRITE--")
RynLib.bind_volumes()
