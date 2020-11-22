
#ifndef RYNLIB_FFIPARAMETERS_HPP
#define RYNLIB_FFIPARAMETERS_HPP

#include "PyAllUp.hpp"
#include <string>
#include <vector>

namespace rynlib {
    namespace PlzNumbers {

// Set up enum for type mapping
// Must be synchronized with the types on the python side
        enum class FFIType {
            PY_TYPES = 1000,
            UnsignedChar = PY_TYPES + 10,
            Short = PY_TYPES + 20,
            UnsignedShort = PY_TYPES + 21,
            Int = PY_TYPES + 30,
            UnsignedInt = PY_TYPES + 31,
            Long = PY_TYPES + 40,
            UnsignedLong = PY_TYPES + 41,
            LongLong = PY_TYPES + 50,
            UnsignedLongLong = PY_TYPES + 51,
            PySizeT = PY_TYPES + 60,

            Float = PY_TYPES + 70,
            Double = PY_TYPES + 71,

            Bool = PY_TYPES + 80,
            String = PY_TYPES + 90,
            PyObject = PY_TYPES + 100,

            NUMPY_TYPES = 2000,

            NUMPY_Int8 = NUMPY_TYPES + 10,
            NUMPY_UnsignedInt8 = NUMPY_TYPES + 11,
            NUMPY_Int16 = NUMPY_TYPES + 12,
            NUMPY_UnsignedInt16 = NUMPY_TYPES + 13,
            NUMPY_Int32 = NUMPY_TYPES + 14,
            NUMPY_UnsignedInt32 = NUMPY_TYPES + 15,
            NUMPY_Int64 = NUMPY_TYPES + 16,
            NUMPY_UnsignedInt64 = NUMPY_TYPES + 17,

            NUMPY_Float16 = NUMPY_TYPES + 20,
            NUMPY_Float32 = NUMPY_TYPES + 21,
            NUMPY_Float64 = NUMPY_TYPES + 22,
            NUMPY_Float128 = NUMPY_TYPES + 23,

            NUMPY_Bool = NUMPY_TYPES + 30
        };



        class FFIParameter {
            // object that maps onto the python FFI stuff...
            PyObject *py_obj;
            std::string param_key;
            void *param_data; // we void pointer this to make it easier to handle
            FFIType type_char;
            std::vector <size_t> shape_vec; // for holding NumPy data
        public:
            FFIParameter(
                    PyObject *obj,
                    std::string& name,
                    FFIType type,
                    std::vector <size_t>& shape
                    ) : py_obj(obj), param_key(name), param_data(), type_char(type), shape_vec(shape) {};

            FFIParameter(PyObject *obj) : py_obj(obj) { init(); }

            void init();

            template <typename T>
            T get_data();

        };

//        python::from_python<FFIParams>;

        class FFIParameters {
            // object that maps onto the python FFI stuff...
            PyObject *py_obj;
            std::vector<FFIParameter> params;
        public:
            FFIParameters(PyObject* param_obj) : py_obj(param_obj) {init();}
            void init();
            FFIParameter get_param(std::string key);

        };

    }
}

#endif //RYNLIB_FFIPARAMETERS_HPP
