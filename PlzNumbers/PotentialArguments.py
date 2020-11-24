"""
Supports the type mapping info necessary to support the kind of FFI stuff we want to do
"""

from .FFI import FFIArgument

__all__ = [
    "PotentialArguments"
]

class PotentialArguments:

    def __init__(self, *args, **kwargs):
        if len(kwargs) == 0:
            self.arg_vec = None
        else:
            self.arg_vec = [self.format_argument(name, a) for name, a in kwargs.items()] # to support old-style potentials
        # supported extra types
        extra_bools = []
        extra_ints = []
        extra_floats = []
        for a in args:
            if a is True or a is False:
                extra_bools.append(a)
            elif isinstance(a, int):
                extra_ints.append(a)
            elif isinstance(a, float):
                extra_floats.append(a)

        self.extra_bools=extra_bools
        self.extra_ints=extra_ints
        self.extra_floats=extra_floats

    @classmethod
    def format_argument(cls, name, val):
        return FFIArgument(
            arg_name=name,
            arg_type=cls.infer_type_code(val),
            arg_shape=cls.get_arg_shape(val)
        )
    @classmethod
    def infer_type_code(cls, val):
        raise NotImplementedError("haven't needed this yet...")
    @classmethod
    def get_arg_shape(cls, val):
        raise NotImplementedError("haven't needed this yet...")

    @property
    def ffi_parameters(self):
        if self.arg_vec is None:
            raise ValueError("potential requires name mapping")
        else:
            return self.arg_vec



