"""
Defines a general potential class that makes use of the PotentialCaller and PotentialLoader
"""

import os
from ..RynUtils import ParameterManager

from .PotentialLoader import PotentialLoader
from .PotentialCaller import PotentialCaller
from .PotentialTemplator import PotentialTemplate
from .PotentialArguments import AtomsPattern, PotentialArgumentSpec

from .FFI import FFIModule

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

    __props__ = [
        'atoms',
        'working_directory'
    ]
    def __init__(self,
                 name,
                 argspec,
                 caller,
                 atoms=None,
                 working_directory=None
                 ):
        """
        :param argspec:
        :type argspec: PotentialArgumentSpec
        :param loader:
        :type loader: PotentialLoader
        """
        self.name = name
        self._argspec = PotentialArgumentSpec(argspec) if not isinstance(argspec, PotentialArgumentSpec) else argspec
        self._caller = caller

        self.working_directory = working_directory
        self._atomspec = AtomsPattern(atoms)
        self._atoms = None # a bound set of atoms
        self._args = None # the bound version of the args

    @classmethod
    def from_options(cls,
                     name=None,
                     potential_source=None,
                     potential_directory=None,
                     function_name=None,
                     arguments=None,
                     wrap_potential=False,
                     **params
                     ):
        """
        Constructs a Potential object from the various options that can be passed to a
        PotentialCaller, Loader, Template, etc.

        :param name:
        :type name:
        :param potential_source:
        :type potential_source:
        :param wrap_potential:
        :type wrap_potential:
        :param params:
        :type params:
        :return:
        :rtype:
        """

        src = potential_source

        params = ParameterManager(**params)


        if potential_directory is None:
            from ..Interface import RynLib
            potential_directory = RynLib.potential_directory()

        if wrap_potential:
            src = cls.wrap_potential(
                name, src,
                potential_directory=potential_directory,
                arguments=arguments,
                **params.filter(PotentialTemplate)
            )

        # prepare args for the loader
        loader = PotentialLoader(
            name,
            src,
            load_path=[os.path.join(potential_directory, name), src],
            **params.filter(PotentialLoader)
        )

        # set up spec
        callable = loader.call_obj
        if isinstance(callable, FFIModule):
            spec = callable.get_method(function_name)
        else:
            spec = PotentialArgumentSpec(arguments, name=function_name)

        # then set up caller
        caller = PotentialCaller(
            callable,
            function_name,
            **params.filter(PotentialCaller)
        )

        return cls(name, spec, caller, **params.filter(cls))

    def __call__(self, coordinates, *extra_args, **extra_kwargs):
        """
        Provides a caller into the potential

        :param coordinates:
        :type coordinates:
        :param extra_args:
        :type extra_args:
        :param extra_kwargs:
        :type extra_kwargs:
        :return:
        :rtype:
        """

        if self._atomspec is not None:
            atoms = self._atoms
        elif len(extra_args) > 0:
            atoms = extra_args[0]
            extra_args = extra_args[1:]
        else:
            atoms = []

        if atoms is not self._atoms:
            atoms = self._atomspec.validate(atoms)

        if len(extra_args) == 0 and len(extra_kwargs) == 0: # no args at all
            extra_args = self._args if self._args is not None else self._argspec.collect_args()
        else:
            extra_args = self._argspec.collect_args(*extra_args, **extra_kwargs)

        if self.working_directory is not None:
            curdir = os.getcwd()
            try:
                os.chdir(self.working_directory)
                pot = self.caller(coordinates, atoms, extra_args)
            finally:
                os.chdir(curdir)
        else:
            pot = self.caller(coordinates, atoms, extra_args)
        return pot

    @property
    def caller(self):
        return self._caller
    @property
    def spec(self):
        return self._argspec
    @property
    def function_name(self):
        return self.spec.name

    def __repr__(self):
        arg_names = self._argspec.arg_names(excluded=["coords", "raw_coords", "atoms"])
        return "Potential('{}', {}({}), atoms={})".format(
            self.name,
            self.function_name,
            ", ".join(arg_names),
            self._atoms
            # self._args
        )

    @property
    def mpi_manager(self):
        return self.caller.mpi_manager
    @mpi_manager.setter
    def mpi_manager(self, manager):
        self.caller.mpi_manager = manager

    def clean_up(self):
        self.caller.clean_up()

    def bind_arguments(self, *args, **kwargs):
        if len(kwargs) == 0 and len(args) == 1: #  for backwards compat.
            if isinstance(args[0], tuple): # got a tuple of args instead of it being unpacked
                self.bind_arguments(*args[0])
            elif isinstance(args[0], dict): # got a dict of args
                self.bind_arguments(**args[0])
        else:
            args = self._argspec.collect_args(*args, **kwargs)
            self._args = args

    def bind_atoms(self, atoms):
        self._atoms = self._atomspec.validate(atoms)

    @classmethod
    def wrap_potential(cls,
                       name,
                       src,
                       potential_directory = None,
                       caller_api_version = 0,
                       **kwargs
                       ):
        if caller_api_version != 1:
            raise ValueError((
                "{}.{} is intended to wrap old-style (single pointer) potentials. "
                "If you want that form of potential, set `caller_api_version=1`. "
                "Otherwise you will be better off using tools from the FFI module."
            ).format(cls.__name__, 'wrap_potential'))
        #
        # if potential_directory is None:
        #     from ..Interface import RynLib
        #     potential_directory = RynLib.potential_directory()
        if not os.path.exists(potential_directory):
            os.makedirs(potential_directory)

        pot_src = src
        src = os.path.join(potential_directory, name)
        if not os.path.exists(os.path.join(src, "src")):
            PotentialTemplate(
                lib_name=name,
                potential_source=pot_src,
                **kwargs
            ).apply(potential_directory)
        return src
