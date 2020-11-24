
import enum
from collections import namedtuple

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
    type_map["b"] = UnsignedChar
    Short = PY_TYPES + 20
    type_map["h"] = Short
    UnsignedShort = PY_TYPES + 21
    type_map["H"] = UnsignedShort
    Int = PY_TYPES + 30
    type_map["i"] = Int
    UnsignedInt = PY_TYPES + 31
    type_map["I"] = UnsignedInt
    Long = PY_TYPES + 40
    type_map["l"] = Long
    UnsignedLong = PY_TYPES + 41
    type_map["L"] = UnsignedLong
    LongLong = PY_TYPES + 50
    type_map["k"] = LongLong
    UnsignedLongLong = PY_TYPES + 51
    type_map["K"] = UnsignedLongLong
    PySizeT = PY_TYPES + 60
    type_map["n"] = PySizeT

    Float = PY_TYPES + 70
    type_map["f"] = Float
    Double = PY_TYPES + 71
    type_map["d"] = Double

    Bool = PY_TYPES + 80
    type_map["p"] = Bool
    String = PY_TYPES + 90
    type_map["s"] = String
    PyObject = PY_TYPES + 100
    type_map["O"] = PyObject

    # supports the NumPy NPY_TYPES enum
    # 200 is python space
    NUMPY_TYPES = 2000

    NUMPY_Int8 = NUMPY_TYPES + 10
    type_map["np.int8"] = NUMPY_Int8
    NUMPY_UnsignedInt8 = NUMPY_TYPES + 11
    type_map["np.uint8"] = NUMPY_UnsignedInt8
    NUMPY_Int16 = NUMPY_TYPES + 12
    type_map["np.int16"] = NUMPY_Int16
    NUMPY_UnsignedInt16 = NUMPY_TYPES + 13
    type_map["np.uint16"] = NUMPY_UnsignedInt16
    NUMPY_Int32 = NUMPY_TYPES + 14
    type_map["np.int32"] = NUMPY_Int32
    NUMPY_UnsignedInt32 = NUMPY_TYPES + 15
    type_map["np.uint32"] = NUMPY_UnsignedInt32
    NUMPY_Int64 = NUMPY_TYPES + 16
    type_map["np.int64"] = NUMPY_Int64
    NUMPY_UnsignedInt64 = NUMPY_TYPES + 17
    type_map["np.uint64"] = NUMPY_UnsignedInt64

    NUMPY_Float16 = PY_TYPES + 20
    type_map["np.float16"] = NUMPY_Float16
    NUMPY_Float32 = PY_TYPES + 21
    type_map["np.float32"] = NUMPY_Float32
    NUMPY_Float64 = PY_TYPES + 22
    type_map["np.float64"] = NUMPY_Float64
    NUMPY_Float128 = PY_TYPES + 23
    type_map["np.float128"] = NUMPY_Float128

    NUMPY_Bool = NUMPY_TYPES + 30
    type_map["np.bool"] = NUMPY_Bool

class FFIArgument:
    """
    An argument spec for data to be passed to an FFIMethod
    """
    def __init__(self, arg_name, arg_type, arg_shape):
        self.arg_name = arg_name
        self.arg_type = FFIType(arg_type)
        self.arg_shape = arg_shape
    def __repr__(self):
        return "{}('{}', {}, {})".format(
            type(self).__name__,
            self.arg_name,
            self.arg_type,
            self.arg_shape
        )

class FFIMethod:
    """
    Represents a C++ method callable through the plzffi interface
    """
    def __init__(self, sig):
        name, args, ret = sig
        self.name = name
        self.args = [FFIArgument(*x) for x in args]
        self.ret = FFIType(ret)

    def __repr__(self):
        return "{}('{}', {})=>{}".format(
            type(self).__name__,
            self.name,
            self.args,
            self.ret
        )

class FFIModule:
    """
    Provides a layer to ingest a Python module containing an '_FFIModule' capsule.
    The capsule is expected to point to a `plzffi::FFIModule` object and can be called using a `PotentialCaller`
    """

    def __init__(self, module):
        self.mod = module
        self.name, self.cap = module._FFIModule
        self._sig = None

    @property
    def signature(self):
        if self._sig is None:
            self._sig = self.mod.get_signature(self.mod._FFIModule)
        return self._sig

    @property
    def method_names(self):
        return tuple(x[0] for x in self.signature[1])

    def __getattr__(self, item):
        try:
            idx = self.method_names.index(item)
        except IndexError:
            idx = None
        if idx is not None:
            return FFIMethod(self.signature[1][idx])
        else:
            raise

    def __repr__(self):
        return "{}('{}', methods={})".format(
            type(self).__name__,
            self.name,
            self.method_names
        )