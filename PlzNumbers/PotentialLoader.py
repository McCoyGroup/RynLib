"""
Provides a Loader object to load a potential from a C++ extension
"""

import os, numpy as np
from ..RynUtils import CLoader, ModuleLoader
from .PotentialCaller import PotentialCaller
from .FFI import FFIModule

__all__ = [
    "PotentialLoader"
]

class PotentialLoader:
    """
    Provides a standardized way to load and compile a potential using a potential template
    """
    __props__ = [
        "src_ext",
        "description",
        "version",
        "include_dirs",
        "linked_libs",
        "macros",
        "source_files",
        "build_script",
        "requires_make",
        "out_dir",
        "cleanup_build",
        'build_kwargs'
    ]

    src_folder = os.path.join(os.path.dirname(__file__), "src")
    libs_folder = os.path.join(os.path.dirname(__file__), "libs")
    cpp_std = '-std=c++17'
    def __init__(self,
                 name,
                 src,
                 src_ext='src',
                 load_path=None,
                 description="A compiled potential",
                 version="1.0.0",
                 include_dirs=None,
                 linked_libs=None,
                 macros=None,
                 source_files=None,
                 build_script=None,
                 requires_make=False,
                 out_dir=None,
                 python_potential=False,
                 cleanup_build=True,
                 pointer_name=None,
                 build_kwargs=None
                 ):
        self.python_potential = python_potential
        # if python_potential is False:

        if include_dirs is None:
            include_dirs = []
        include_dirs = (
                               tuple(include_dirs)
                               + (self.src_folder, self.libs_folder, np.get_include())
                               + (PotentialCaller.TBB_CentOS, PotentialCaller.TBB_Ubutu)
        )
        if linked_libs is None:
            linked_libs = []
        linked_libs = tuple(linked_libs) + ("plzffi", 'tbb', 'tbbmalloc', 'tbbmalloc_proxy')

        self.c_loader = CLoader(
                              name, src,
                              load_path=load_path,
                              src_ext=src_ext,
                              description=description,
                              version=version,
                              include_dirs=include_dirs,
                              linked_libs=linked_libs,
                              macros=macros,
                              source_files=source_files,
                              build_script=build_script,
                              requires_make=requires_make,
                              out_dir=out_dir,
                              cleanup_build=cleanup_build,
                              extra_compile_args=["-fopenmp", self.cpp_std],
                              extra_link_args=["-fopenmp"],
                              **({} if build_kwargs is None else build_kwargs)
                              )
        # else:
        #     self.c_loader = None

        # Need to insert code here to allow for new caller API to work
        self._lib = None

        self._attr = pointer_name
        # self.function_name = pointer_name

    @property
    def lib(self):
        if self._lib is None:
            if self.python_potential is True:
                loader = ModuleLoader()
                remade = False
                try:
                    self._lib = loader.load(self.c_loader.lib_dir, "")#, self.c_loader.lib_name+"Lib")
                except ImportError:
                    if self.c_loader.requires_make:
                        remade = True
                        self.c_loader.make_required_libs()
                    else:
                        raise
                if remade:
                    self._lib = loader.load(self.c_loader.lib_dir, "")#self.c_loader.lib_name + "Lib")
            elif self.python_potential is False:
                self._lib = self.c_loader.load()
        return self._lib
    @property
    def caller_api_version(self):
        if hasattr(self.lib, "_FFIModule"): # currently how we're dispatching
            return 2
        else:
            return 1
    @property
    def call_obj(self):
        """
        The object that defines how to call the potential.
        Can either be a pure python function, an FFIModule, or a PyCapsule

        :return:
        :rtype:
        """
        if self.python_potential is not False and self.python_potential is not True:
            return self.python_potential
        else:
            if hasattr(self.lib, "_FFIModule"):
                return FFIModule.from_module(self._lib)
            else:
                return self._lib._potential
            # return getattr(self.lib, self._attr)