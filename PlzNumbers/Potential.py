"""
Defines a general potential class that makes use of the PotentialCaller and PotentialLoader
"""

from .PotentialLoader import PotentialLoader
from .PotentialCaller import PotentialCaller
from .PotentialTemplator import PotentialTemplate
import os

__all__ = [
    "Potential"
]

class Potential:
    """
    A very general wrapper to a potential:
        Can take a potential _directory_ and compile that down
        Can take a potential source and write the necessary template code around that for use
    Provides a hook into PotentialCaller once the data has been loaded to directly call the potential like a function
    """
    def __init__(self,
                 name,
                 src,
                 *ignore,

                 #Template Options
                 wrap_potential = None,
                 function_name=None,
                 raw_array_potential=False,
                 arguments=(),
                 potential_directory = None,

                 #Loader Options
                 src_ext='src',
                 description="An extension module",
                 verion="1.0.0",
                 include_dirs=None,
                 linked_libs=None,
                 macros=None,
                 source_files=None,
                 build_script=None,
                 requires_make=False,
                 out_dir=None,
                 cleanup_build=True,

                 #Caller Options
                 bad_walker_file="bad_walkers.txt",
                 mpi_manager=None,
                 vectorized_potential=False,
                 error_value=10.e9
                 ):
        self.name = name

        if wrap_potential:
            if potential_directory is None:
                potential_directory = os.path.expanduser("~/RynLib/potentials")
            if not os.path.exists(potential_directory):
                os.makedirs(potential_directory)
            pot_src = src
            src = os.path.join(potential_directory, name)
            if not os.path.exists(src):
                PotentialTemplate(
                    lib_name=name,
                    potential_source=pot_src,
                    function_name=function_name,
                    raw_array_potential=raw_array_potential,
                    arguments=arguments
                ).apply(potential_directory)

        self.src = src

        self.loader = PotentialLoader(
            name,
            src,
            src_ext=src_ext,
            description=description,
            verion=verion,
            include_dirs=include_dirs,
            linked_libs=linked_libs if linked_libs is not None else [name],
            macros=macros,
            source_files=source_files,
            build_script=build_script,
            requires_make=requires_make,
            out_dir=out_dir,
            cleanup_build=cleanup_build
        )

        self._caller = None
        self._caller_opts = dict(
            bad_walker_file=bad_walker_file,
            mpi_manager=mpi_manager,
            raw_array_potential=raw_array_potential,
            vectorized_potential=vectorized_potential,
            error_value=error_value
        )

    @property
    def caller(self):
        if self._caller is None:
            self._caller = PotentialCaller(
                self.loader.pointer,
                **self._caller_opts
            )
        return self._caller

    def __call__(self, coordinates, atoms, *extra_args):
        return self.caller(coordinates, atoms, *extra_args)