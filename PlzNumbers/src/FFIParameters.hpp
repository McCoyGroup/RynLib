
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

        //
        template <typename T>
        class FFITypeHandler {
            // class that helps us maintain a map between type codes & proper types
        public:
            void validate(FFIType type_code);
            T cast(FFIType type_code, void* data);
        };
        template <typename T>
        class FFITypeHandler<T*> {
            // specialization to handle pointer types
        public:
            void validate(FFIType type_code);
            T* cast(FFIType type_code, void* data);
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

            explicit FFIParameter(PyObject *obj) : py_obj(obj) { init(); }

            void init();

            std::string name() { return param_key; }

            template <typename T>
            T value();

            void* _raw_ptr() { return param_data; } // I put this out there so people smarter than I can use it

        };

        class FFIParameters {
            // object that maps onto the python FFI stuff...
            PyObject *py_obj;
            std::vector<FFIParameter> params;
        public:
            FFIParameters() : py_obj(), params() {};
            explicit FFIParameters(PyObject* param_obj) : py_obj(param_obj) {
                params = {};
                init();
            }
            void init();
            int param_index(std::string& key);
            FFIParameter get_parameter(std::string& key);
            FFIParameter set_parameter(std::string& key, FFIParameter& param);

        };

    }
}

#endif //RYNLIB_FFIPARAMETERS_HPP
