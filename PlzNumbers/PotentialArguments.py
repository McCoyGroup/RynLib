
from .FFI import FFIMethod

__all__ = [
    "PotentialArguments"
]

class PotentialArguments:
    """
    Simple wrapper to support both old- and new-style calling semantics
    """

    def __init__(self, potential_function, *args, **kwargs):
        self.pot_func = potential_function
        if isinstance(potential_function, FFIMethod):
            self.arg_vec = FFIMethod.collect_args(*args, **kwargs)
        else: # to support old-style potentials
            self.arg_vec = None
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

            self.extra_bools = extra_bools
            self.extra_ints = extra_ints
            self.extra_floats = extra_floats

    @property
    def ffi_parameters(self):
        if self.arg_vec is None:

            raise ValueError("potential requires parameters as a dictionary")
        else:
            return self.arg_vec.values()



