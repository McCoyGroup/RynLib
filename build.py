"""
Top-level build file that will precompile the various portions of the project that should be precompiled in a
containerized environment
"""

from .Interface import GeneralConfig

print("--INSTALLING OPEN MPI--")
GeneralConfig.install_MPI()

print("--MOUNTING VOLUMES FOR DATA READ/WRITE--")
GeneralConfig.bind_volumes()
