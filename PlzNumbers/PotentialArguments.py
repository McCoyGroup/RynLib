
from collections import OrderedDict
from .FFI import FFIMethod

__all__ = [
    "PotentialArguments"
]

class PotentialArgumentHolder:
    """
    Wrapper class that simply holds onto
    """
    def __init__(self, args):
        if isinstance(args, OrderedDict):
            self.arg_vec = args
        else:
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
            raise ValueError("Python thinks we're using an old-style potential")
        else:
            return self.arg_vec.values()
    def __repr__(self):
        return "{}(<>)".format(type(self).__name__)

class OldStyleArg:
    """
    Shim that supports old-style args
    """
    def __init__(self, name=None, dtype=None, extra=None, shape=None, default=None):
        self.name=name
        self.dtype=self.canonicalize_dtype(dtype)
        self.extra=extra
        self.shape=shape
        if default is not None:
            raise ValueError("currrently, no longer supporting default values")
        self.default=default
    _type_map = {'int':int, 'float':float, 'str':str}
    def canonicalize_dtype(self, dtype):
        if isinstance(dtype, str):
            return self._type_map[dtype]
        else:
            raise ValueError("don't know how to handle old-style dtype '{}'".format(dtype))
    def cast(self, v):
        if not isinstance(v, self.dtype):
            raise ValueError(
                "Argument mismatch: argument '{}' is expected to be of type {} (got {})".format(
                    self.name,
                    self.dtype.__name__,
                    type(v).__name__
                ))
        return v
    @property
    def arg_name(self):
        return self.name

class OldStyleArgList:
    """
    Shim that supports old-style arg lists
    """
    def __init__(self, args):
        self.args = [OldStyleArg(**a) for a in args]

    @property
    def arg_names(self):
        return tuple(a.arg_name for a in self.args)

    def collect_args(self, *args, excluded_args=None, **kwargs):
        arg_dict = OrderedDict()
        req_dict = OrderedDict(
            (k.arg_name, k) for k in self.args
        )
        for k in kwargs:
            arg_dict[k] = req_dict[k].cast(kwargs[k])
            del req_dict[k]

        if excluded_args is not None:
            for k in excluded_args:
                if k in req_dict:
                    del req_dict[k]

        if len(req_dict) > 0:
            for v, k in zip(args, req_dict.copy()):
                arg_dict[k] = req_dict[k].cast(v)
                del req_dict[k]
        if len(req_dict) > 0:
            raise ValueError("{}.{}: missing required arguments {}".format(
                type(self).__name__,
                'collect_args',
                tuple(req_dict.values())
            ))

        return tuple(arg_dict.values()) # need this to tell PotentialArgumentHolder that this is old-style

class PotentialArgumentSpec:
    """
    Simple wrapper to support both old- and new-style calling semantics
    """
    def __init__(self, arg_pattern, name=None):
        self._name = name
        self.arg_pat = self.canonicalize_pats(arg_pattern)
    @property
    def name(self):
        if self._name is None:
            return self.arg_pat.name
        else:
            return self._name
            

    def canonicalize_pats(self, pats):
        if isinstance(pats, FFIMethod):
            return pats

        arg_list = []
        for a in pats:
            if isinstance(a, (dict, OrderedDict)):
                name = a['name']
                dtype = a['dtype']
                default = a['default'] if 'default' in a else None
                shape = a['shape'] if 'shape' in a else ()
            else:
                name = a[0]
                dtype = a[1]
                default = a[2] if len(a) > 2 in a else None
                shape = a[3] if len(a) > 3 in a else ()
            arg_list.append({
                "name":name,
                "dtype":dtype,
                "default":default,
                "shape":shape
            })
        return OldStyleArgList(arg_list)

    def collect_args(self, *args, **kwargs):
        return PotentialArgumentHolder(self.arg_pat.collect_args(*args, excluded_args=["coords", "raw_coords", "atoms"], **kwargs))

    def arg_names(self, excluded=None):
        """
        Canonicalizes arguments, if passed
        :return:
        :rtype:
        """

        names = self.arg_pat.arg_names
        if excluded is None:
            return tuple(x for x in names if x not in excluded)
        else:
            return names

class AtomsPattern:
    """
    Spec to define a pattern for atoms so that calls can be validated before a function
    is every called
    """
    def __init__(self, atoms):
        self.atom_pat = atoms

    def validate(self, atoms):
        if self.atom_pat is not None:
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
        return atoms


