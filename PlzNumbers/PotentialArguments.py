"""
Supports the type mapping info necessary to support the kind of FFI stuff we want to do
"""

__all__ = [
    "PotentialArguments"
]

from collections import namedtuple
import enum

PotentialArgument = namedtuple(
    "PotentialArgument",
    ["dtype", "shape", "target_dtype"]
)

# BoolType = PrimitiveType(
#     "bool",
#     ctypes.c_bool,
#     "bool",
#     "p",
#     (bool,),
#     (np.dtype('bool'),),
#     None,#serializer
#     None#deserializer
# )
# # C-types with the same names
# FloatType = PrimitiveType(
#     "float",
#     ctypes.c_float,
#     "float",
#     "f",
#     (float,),
#     (np.dtype('float32'),),
#     None,#serializer
#     None#deserializer
# )
# DoubleType = PrimitiveType(
#     "double",
#     ctypes.c_double,
#     "float",
#     "d",
#     (float,),
#     (np.dtype('float64'),),
#     None,#serializer
#     None#deserializer
# )
# IntType = PrimitiveType(
#     "int",
#     ctypes.c_int,
#     "int",
#     "i",
#     (int,),
#     (np.dtype('int32'),),
#     None,#serializer
#     None#deserializer
# )
# LongType = PrimitiveType(
#     "long",
#     ctypes.c_int,
#     "long",
#     "i",
#     (int,),
#     (np.dtype('int32'),),
#     None,#serializer
#     None#deserializer
# )

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

    UnsignedChar = PY_TYPES.value + 10
    type_map["b"] = UnsignedChar
    Short = PY_TYPES.value + 20
    type_map["h"] = Short
    UnsignedShort = PY_TYPES.value + 21
    type_map["H"] = UnsignedShort
    Int = PY_TYPES.value + 30
    type_map["i"] = Int
    UnsignedInt = PY_TYPES.value + 31
    type_map["I"] = UnsignedInt
    Long = PY_TYPES.value + 40
    type_map["l"] = Long
    UnsignedLong = PY_TYPES.value + 41
    type_map["L"] = UnsignedLong
    LongLong = PY_TYPES.value + 50
    type_map["k"] = LongLong
    UnsignedLongLong = PY_TYPES.value + 51
    type_map["K"] = UnsignedLongLong
    PySizeT = PY_TYPES.value + 60
    type_map["n"] = PySizeT

    Float = PY_TYPES.value + 70
    type_map["f"] = Float
    Double = PY_TYPES.value + 71
    type_map["d"] = Double

    Bool = PY_TYPES.value + 80
    type_map["p"] = Bool
    String = PY_TYPES.value + 90
    type_map["s"] = String
    PyObject = PY_TYPES.value + 100
    type_map["O"] = PyObject

    # supports the NumPy NPY_TYPES enum
    # 200 is python space
    NUMPY_TYPES = 2000

    NUMPY_Int8 = NUMPY_TYPES.value + 10
    type_map["np.int8"] = NUMPY_Int8
    NUMPY_UnsignedInt8 = NUMPY_TYPES.value + 11
    type_map["np.uint8"] = NUMPY_UnsignedInt8
    NUMPY_Int16 = NUMPY_TYPES.value + 12
    type_map["np.int16"] = NUMPY_Int16
    NUMPY_UnsignedInt16 = NUMPY_TYPES.value + 13
    type_map["np.uint16"] = NUMPY_UnsignedInt16
    NUMPY_Int32 = NUMPY_TYPES.value + 14
    type_map["np.int32"] = NUMPY_Int32
    NUMPY_UnsignedInt32 = NUMPY_TYPES.value + 15
    type_map["np.uint32"] = NUMPY_UnsignedInt32
    NUMPY_Int64 = NUMPY_TYPES.value + 16
    type_map["np.int64"] = NUMPY_Int64
    NUMPY_UnsignedInt64 = NUMPY_TYPES.value + 17
    type_map["np.uint64"] = NUMPY_UnsignedInt64

    NUMPY_Float16 = PY_TYPES.value + 20
    type_map["np.float16"] = NUMPY_Float16
    NUMPY_Float32 = PY_TYPES.value + 21
    type_map["np.float32"] = NUMPY_Float32
    NUMPY_Float64 = PY_TYPES.value + 22
    type_map["np.float64"] = NUMPY_Float64
    NUMPY_Float128 = PY_TYPES.value + 23
    type_map["np.float128"] = NUMPY_Float128

    NUMPY_Bool = NUMPY_TYPES.value + 30
    type_map["np.bool"] = NUMPY_Bool

class PotentialArguments:

    def __init__(self, *args):
        self.arg_vec = [self.format_argument(a) for a in args]

