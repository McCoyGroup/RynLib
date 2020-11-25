
import enum, numpy as np, collections

__all__ = [
    "FFIModule",
    "FFIMethod",
    "FFIArgument",
    "FFIType",
    "FFIModuleSpec",
    "FFIMethodSpec"
]

class FFIType(enum.Enum):
    """
    The set of supported enum types.
    Maps onto the native python convertable types and NumPy dtypes.
    In the future, this should be done more elegantly, but for now it suffices
    that these types align on the C++ side and this side.
    Only NumPy arrays are handled using the buffer interface & so if you want to pass a pointer
    you gotta do it using a NumPy array.
    """

    type_map = {}

    PY_TYPES=1000

    UnsignedChar = PY_TYPES + 10
    type_map[UnsignedChar] = ("b", int)
    Short = PY_TYPES + 20
    type_map[Short] = ("h", int)
    UnsignedShort = PY_TYPES + 21
    type_map[UnsignedShort] = ("H", int)
    Int = PY_TYPES + 30
    type_map[Int] = ("i", int)
    UnsignedInt = PY_TYPES + 31
    type_map["I"] = UnsignedInt
    type_map[UnsignedInt] = ("I", int)
    Long = PY_TYPES + 40
    type_map[Long] = ("l", int)
    UnsignedLong = PY_TYPES + 41
    type_map[UnsignedLong] = ("L", int)
    LongLong = PY_TYPES + 50
    type_map["k"] = LongLong
    type_map[LongLong] = ("k", int)
    UnsignedLongLong = PY_TYPES + 51
    type_map["K"] = UnsignedLongLong
    type_map[UnsignedLongLong] = ("K", int)
    PySizeT = PY_TYPES + 60
    type_map[PySizeT] = ("n", int)

    Float = PY_TYPES + 70
    type_map["f"] = Float
    type_map[Float] = ("f", float)
    Double = PY_TYPES + 71
    type_map["d"] = Double
    type_map[Double] = ("d", float)

    Bool = PY_TYPES + 80
    type_map[Bool] = ("p", bool)
    String = PY_TYPES + 90
    type_map[String] = ("s", str)
    PyObject = PY_TYPES + 100
    type_map["O"] = PyObject
    type_map[PyObject] = ("O", object)

    # supports the NumPy NPY_TYPES enum
    # 200 is python space
    NUMPY_TYPES = 2000

    NUMPY_Int8 = NUMPY_TYPES + 10
    type_map[NUMPY_Int8] = ("np.int8", np.int8)
    NUMPY_UnsignedInt8 = NUMPY_TYPES + 11
    type_map[NUMPY_UnsignedInt8] = ("np.uint8", np.uint8)
    NUMPY_Int16 = NUMPY_TYPES + 12
    type_map[NUMPY_Int16] = ("np.int16", np.int16)
    NUMPY_UnsignedInt16 = NUMPY_TYPES + 13
    type_map[NUMPY_UnsignedInt16] = ("np.uint16", np.uint16)
    NUMPY_Int32 = NUMPY_TYPES + 14
    type_map[NUMPY_Int32] = ("np.int32", np.int32)
    NUMPY_UnsignedInt32 = NUMPY_TYPES + 15
    type_map["np.uint32"] = NUMPY_UnsignedInt32
    type_map[NUMPY_UnsignedInt32] = ("np.uint32", np.uint32)
    NUMPY_Int64 = NUMPY_TYPES + 16
    type_map[NUMPY_Int64] = ("np.int64", np.int64)
    NUMPY_UnsignedInt64 = NUMPY_TYPES + 17
    type_map[NUMPY_UnsignedInt64] = ("np.uint64", np.uint64)

    NUMPY_Float16 = PY_TYPES + 20
    type_map[NUMPY_Float16] = ("np.float16", np.float16)
    NUMPY_Float32 = PY_TYPES + 21
    type_map[NUMPY_Float32] = ("np.float32", np.float32)
    NUMPY_Float64 = PY_TYPES + 22
    type_map[NUMPY_Float64] = ("np.float64", np.float64)
    NUMPY_Float128 = PY_TYPES + 23
    type_map[NUMPY_Float128] = ("np.float128", np.float128)

    NUMPY_Bool = NUMPY_TYPES + 30
    type_map[NUMPY_Bool] = ("np.bool", np.bool)


class FFISpec:
    """
    Provides a uniform layout for handling specs of different parts of an FFI library
    """
    __fields__ = []

    def __init__(self, **kwargs):
        for k in self.__fields__:
            if k not in kwargs or kwargs[k] is None:
                raise ValueError("{} got no value for required parameter {}".format(
                    type(self).__name__,
                    k
                ))
        for k in kwargs:
            if k not in self.__fields__:
                raise ValueError("{} got value for unexpected parameter {}".format(
                    type(self).__name__,
                    k
                ))
class FFIArgument(FFISpec):
    """
    An argument spec for data to be passed to an FFIMethod
    """
    __fields__ = ["name", "dtype", "shape"]
    def __init__(self, name=None, dtype=None, shape=None):
        if shape is None:
            shape = ()
        super().__init__(name=name, dtype=dtype, shape=shape)
        self.arg_name = name
        self.arg_type = FFIType(dtype)
        self.arg_shape = shape
    def __repr__(self):
        return "{}('{}', {}, {})".format(
            type(self).__name__,
            self.arg_name,
            self.arg_type,
            self.arg_shape
        )
    def cast(self, val):
        type_str, dtype = FFIType.type_map[self.arg_type]
        if isinstance(dtype, np.dtype): # have a numpy type
            dat = np.asarray(val, dtype=dtype)
        else:
            if not isinstance(val, dtype):
                dat = dtype(val)
            else:
                dat = val
        # do some shape checks...
        return dat

class FFIMethod(FFISpec):
    """
    Represents a C++ method callable through the plzffi interface
    """
    __fields__ = ["name", "arguments", "rtype"]
    def __init__(self, name=None, arguments=None, rtype=None, module=None):
        super().__init__(name=name, arguments=arguments, rtype=rtype)
        self.name = name
        self.args = [FFIArgument(**x) for x in arguments]
        self.rtype = FFIType(rtype)
        self.mod = module

    def collect_args(self, *args, **kwargs):
        arg_dict = collections.OrderedDict()
        req_dict = collections.OrderedDict(
            (k.arg_name, k) for k in self.args
        )
        for k in kwargs:
            arg_dict[k] = req_dict[k].cast(kwargs[k])
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
        return arg_dict

    @classmethod
    def from_signature(cls, sig, module=None):
        name, args, ret = sig
        return cls(
            name=name,
            arguments=[{k:v for k,v in zip(FFIArgument.__fields__, x)} for x in args],
            rtype=ret,
            module=module
        )
    def __repr__(self):
        return "{}('{}', {})=>{}".format(
            type(self).__name__,
            self.name,
            self.args,
            self.rtype
        )

class FFIModule(FFISpec):
    """
    Provides a layer to ingest a Python module containing an '_FFIModule' capsule.
    The capsule is expected to point to a `plzffi::FFIModule` object and can be called using a `PotentialCaller`
    """
    __fields__ = ["name", "methods"]
    def __init__(self, name=None, methods=None, module=None):
        super().__init__(name=name, methods=methods)
        self.name = name
        self.methods = [FFIMethod(**x) for x in methods]
        self.mod = module

    @classmethod
    def from_signature(cls, sig, module=None):
        name, meths = sig
        return cls(
            name=name,
            methods=[{k:v for k,v in zip(FFIMethod.__fields__, x)} for x in meths],
            module=module
        )

    @classmethod
    def from_module(cls, module):
        name, cap = module._FFIModule
        sig = module.get_signature(module._FFIModule)
        return cls.from_signature(sig, module=module)

    @property
    def method_names(self):
        return tuple(x.name for x in self.methods)

    def get_method(self, name):
        try:
            idx = self.method_names.index(name)
        except (ValueError, IndexError):
            idx = None
        if idx is not None:
            return FFIMethod.from_signature(self.signature[1][idx], module=self)
        else:
            raise AttributeError("FFIModule {} has no method {}".format(self.name, name))
    def __getattr__(self, item):
        return self.get_method(item)

    def __repr__(self):
        return "{}('{}', methods={})".format(
            type(self).__name__,
            self.name,
            self.method_names
        )