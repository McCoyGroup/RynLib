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
                 name = None,
                 potential_source = None,

                 #Validation
                 atom_pattern = None,

                 #Template Options
                 wrap_potential = None,
                 function_name=None,
                 raw_array_potential=None,
                 arguments=None,
                 shim_script="",
                 conversion = None,
                 potential_directory = None,
                 static_source = False,
                 extra_functions=(),

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
                 python_potential=False,
                 pointer_name=None,
                 fortran_potential=False,

                 #Caller Options
                 bad_walker_file="bad_walkers.txt",
                 mpi_manager=None,
                 vectorized_potential=False,
                 error_value=10.e9,
                 transpose_call = None
                 ):
        """

        :param name:
        :type name:
        :param potential_source:
        :type potential_source:
        :param wrap_potential:
        :type wrap_potential:
        :param function_name:
        :type function_name:
        :param raw_array_potential:
        :type raw_array_potential:
        :param arguments:
        :type arguments:
        :param potential_directory:
        :type potential_directory:
        :param static_source:
        :type static_source:
        :param extra_functions:
        :type extra_functions:
        :param src_ext:
        :type src_ext:
        :param description:
        :type description:
        :param verion:
        :type verion:
        :param include_dirs:
        :type include_dirs:
        :param linked_libs:
        :type linked_libs:
        :param macros:
        :type macros:
        :param source_files:
        :type source_files:
        :param build_script:
        :type build_script:
        :param requires_make:
        :type requires_make:
        :param out_dir:
        :type out_dir:
        :param cleanup_build:
        :type cleanup_build:
        :param python_potential:
        :type python_potential:
        :param bad_walker_file:
        :type bad_walker_file:
        :param mpi_manager:
        :type mpi_manager:
        :param vectorized_potential:
        :type vectorized_potential:
        :param error_value:
        :type error_value:
        """
        src = potential_source
        self.name = name
        self.function_name = function_name

        self._atoms = None
        self._args = ()

        if wrap_potential:
            if potential_directory is None:
                from ..Interface import RynLib
                potential_directory = RynLib.potential_directory()
            if not os.path.exists(potential_directory):
                os.makedirs(potential_directory)
            pot_src = src
            src = os.path.join(potential_directory, name)
            if not os.path.exists(os.path.join(src, "src")):
                PotentialTemplate(
                    lib_name=name,
                    potential_source=pot_src,
                    function_name=function_name,
                    raw_array_potential=raw_array_potential,
                    fortran_potential=fortran_potential,
                    shim_script=shim_script,
                    conversion=conversion,
                    arguments=arguments,
                    static_source=static_source,
                    extra_functions=extra_functions
                ).apply(potential_directory)
        self.src = src

        if potential_directory is None:
            from ..Interface import RynLib
            potential_directory = RynLib.potential_directory()
        main_path = os.path.join(potential_directory, name) # I can't remember why I thought we'd want anything else...?

        self.loader = PotentialLoader(
            name,
            src,
            load_path=[main_path, src],
            src_ext=src_ext,
            description=description,
            version=verion,
            include_dirs=include_dirs,
            linked_libs=linked_libs if linked_libs is not None else [name],
            macros=macros,
            source_files=source_files,
            build_script=build_script,
            requires_make=requires_make,
            out_dir=out_dir,
            cleanup_build=cleanup_build,
            python_potential=python_potential,
            pointer_name=pointer_name
        )

        self._caller = None
        self._caller_opts = dict(
            bad_walker_file=bad_walker_file,
            mpi_manager=mpi_manager,
            raw_array_potential=raw_array_potential,
            vectorized_potential=vectorized_potential,
            error_value=error_value,
            fortran_potential=fortran_potential,
            transpose_call = transpose_call
        )


        self._args_pat = arguments
        self._atom_pat = atom_pattern
        self._real_args = None

    def __repr__(self):
        if self._real_args is None:
            self._prep_real_args()
        return "Potential('{}', {}({}), atoms={}, bound_args={})".format(
            self.name,
            self.function_name,
            ", ".join(a[0] for a in self._real_args),
            self._atoms,
            self._args
        )

    @property
    def caller(self):
        if self._caller is None:
            self._caller = PotentialCaller(
                self.loader.pointer,
                **self._caller_opts
            )
        return self._caller
    @property
    def mpi_manager(self):
        return self.caller.mpi_manager
    @mpi_manager.setter
    def mpi_manager(self, manager):
        self.caller.mpi_manager = manager

    def clean_up(self):
        self.caller.clean_up()

    def _validate_atoms(self, atoms):
        if self._atom_pat is not None:
            import re
            if isinstance(self._atom_pat, str):
                self._atom_pat = re.compile(self._atom_pat)
            matches = True
            bad_ind = -1
            try:
                matches = re.match(self._atom_pat, "".join(atoms))
            except TypeError:
                for i,a in enumerate(zip(self._atom_pat, atoms)):
                    a1, a2 = a
                    if a1 != a2:
                        matches = False
                        bad_ind = i
            if not matches and bad_ind >= 0:
                raise ValueError("Atom mismatch at {}: expected atom list {} but got {}".format(
                    bad_ind,
                    tuple(self._atom_pat),
                    tuple(atoms)
                ))
            elif not matches:
                raise ValueError("Atom mismatch: expected atom pattern {} but got list {}".format(
                    bad_ind,
                    self._atom_pat.pattern,
                    tuple(atoms)
                ))
    def bind_atoms(self, atoms):
        self._validate_atoms(atoms)
        self._atoms = atoms
    def _prep_real_args(self):
        self._real_args = []
        if self._args_pat is not None:
            for k in self._args_pat:
                extra = True
                dtype = None
                if isinstance(k, dict):
                    name = k['name']
                    dtype = k['dtype']
                    if 'extra' in k:
                        extra = k['extra']
                    elif name in {"coords", "raw_coords", "atoms", "raw_atoms", "energy"}:
                        extra = False
                    elif isinstance(dtype, str) and dtype.endswith("*"):
                        extra = False
                else:
                    name = k[0]
                    dtype = k[1]
                    if len(k) > 3:
                        extra = k[3]
                    elif k[0] in {"coords", "raw_coords", "atoms", "raw_atoms", "energy"}:
                        extra = False
                    elif isinstance(k[1], str) and k[1].endswith("*"):
                        extra = False

                if extra:
                    if isinstance(dtype, str):
                        dt = dtype.lower()
                        if dt.startswith('float') or dt.startswith('double') or dt == "Real_t":
                            dtype = float
                        elif dt.startswith('int'):
                            dtype = int
                        elif dt == "bool":
                            dtype = bool
                        else:
                            dtype = None
                    self._real_args.append([name, dtype])
    def _validate_args(self, argtuple):
        # first we find the args we actually need...
        if self._args_pat is not None:
            if self._real_args is None:
                self._prep_real_args()
            if isinstance(argtuple, dict):
                args = []
                for k in self._real_args:
                    n = k[0]
                    if n not in argtuple:
                        raise ValueError("Argument mismatch: argument {} missing".format(
                            n
                        ))
                    args.append(argtuple[n])
                argtuple = args
            if len(self._real_args) < len(argtuple):
                raise ValueError("Argument mismatch: too many parameters passed, expected {} but got {}".format(
                    len(self._real_args),
                    len(argtuple)
                ))
            elif len(self._real_args) > len(argtuple):
                raise ValueError("Argument mismatch: too few parameters passed, expected {} but got {}".format(
                    len(self._real_args),
                    len(argtuple)
                ))
            else:
                for i,t in enumerate(zip(self._real_args, argtuple)):
                    t1, obj = t
                    if t1 is not None and not isinstance(obj, t1[1]):
                        raise ValueError("Argument mismatch: argument at {} is expected to be of type {} (got {})".format(
                            i,
                            t1.__name__,
                            type(obj).__name__
                        ))
        return argtuple
    def bind_arguments(self, args):
        args = self._validate_args(args)
        self._args = args
    def __call__(self, coordinates, *extra_args, **extra_kwargs):
        if self._atoms is not None:
            atoms = self._atoms
        elif len(extra_args) > 0:
            atoms = extra_args[0]
            extra_args = extra_args[1:]
        else:
            atoms = []
        if atoms is not self._atoms:
            self._validate_atoms(atoms)
        if len(extra_args) == 0 and len(extra_kwargs) == 0:
            extra_args = self._args
        elif len(extra_args) > 0:
            self._validate_args(extra_args)
        elif len(extra_kwargs) > 0:
            extra_args = self._validate_args(extra_kwargs)
        return self.caller(coordinates, atoms, *extra_args)