
#include "FFIParameters.hpp"
#include "PyAllUp.hpp"
#include <stdexcept>
#include <string>

namespace plzffi {

    using namespace rynlib::python;

    bool DEBUG_PRINT=false;
    bool debug_print() {
        return DEBUG_PRINT;
    }
    void set_debug_print(bool db) {
        DEBUG_PRINT=db;
        pyadeeb.set_debug_print(db); // because of bad design choices I gotta do this multiple places...
    }

    // defines a compiler map between FFIType and proper types
//        template <typename T>
//        void FFITypeHandler<T>::validate(FFIType type_code) {
//            if (type_code != FFIType::GENERIC) {
//                throw std::runtime_error("unhandled type specifier");
//            }
//        };
    // we do the numpy types preferentially because they are most likely to cover the dtype sizes we need
    template <>
    void  FFITypeHandler<npy_bool>::validate(FFIType type_code) {
//            if (type_code != FFIType::UnsignedChar) {
//                throw std::runtime_error("type 'unsigned char'/'npy_bool' misaligned with type code; expected 'UnsignedChar'/'NUMPY_Bool'");
//            };
        if (
                type_code != FFIType::NUMPY_Bool
                && type_code != FFIType::NUMPY_UnsignedInt8
                && type_code != FFIType::UnsignedChar) {
            throw std::runtime_error("type 'unsigned char'/'npy_bool'/'npy_uint8' misaligned with type code; expected 'UnsignedChar'/'NUMPY_Bool'/'NUMPY_UnsignedInt8'");
        };
    }
    template <>
    void  FFITypeHandler<npy_int8>::validate(FFIType type_code) {
        if (type_code != FFIType::NUMPY_Int8) {
            throw std::runtime_error("type 'npy_int8' misaligned with type code; expected 'NUMPY_Int8'");
        };
    }
    template <>
    void  FFITypeHandler<npy_int16>::validate(FFIType type_code) {
        if (type_code != FFIType::Short && type_code != FFIType::NUMPY_Int16) {
            throw std::runtime_error("type 'short'/'npy_int16' misaligned with type code; expected 'Short'/'NUMPY_Int16'");
        };
//            if (type_code != FFIType::NUMPY_Int16) {
//                throw std::runtime_error("type 'npy_int16' misaligned with type code; expected 'NUMPY_Int16'");
//            };
    }
    template <>
    void  FFITypeHandler<npy_int32>::validate(FFIType type_code) {
        if (type_code != FFIType::NUMPY_Int32 && type_code != FFIType::Int) {
            throw std::runtime_error("type 'int'/'npy_int32' misaligned with type code; expected 'Int'/'NUMPY_Int32'");
        };
    }
    template <>
    void  FFITypeHandler<npy_int64>::validate(FFIType type_code) {
        if (type_code != FFIType::NUMPY_Int64 && type_code != FFIType::Long && type_code != FFIType::PySizeT) {
            throw std::runtime_error("type 'long'/'Py_Ssize_t'/'npy_int64' misaligned with type code; expected 'Long'/'PySizeT'/'NUMPY_Int64'");
        };
    }
//        template <>
//        void  FFITypeHandler<npy_uint8>::validate(FFIType type_code) {
//            if (type_code != FFIType::NUMPY_UnsignedInt8) {
//                throw std::runtime_error("type 'npy_uint8' misaligned with type code; expected 'NUMPY_UnsignedInt8'");
//            };
//        }
    template <>
    void  FFITypeHandler<npy_uint16>::validate(FFIType type_code) {
//            if (type_code != FFIType::NUMPY_Float16) {
//                throw std::runtime_error("type 'npy_float16' misaligned with type code; expected 'NUMPY_Float16'");
//            };
        if (
                type_code != FFIType::NUMPY_UnsignedInt16
                && type_code != FFIType::NUMPY_Float16
                && type_code != FFIType::UnsignedShort
                ) {
            throw std::runtime_error(
                    "type 'npy_uint16'/'npy_float16'/'unsigned short' misaligned with type code; expected 'NUMPY_UnsignedInt16'/'NUMPY_Float16'/'UnsignedShort'"
                    );
        };
        if (type_code != FFIType::UnsignedShort) {
            throw std::runtime_error("type 'unsigned short ' misaligned with type code; expected 'UnsignedShort'");
        };
    }
    template <>
    void  FFITypeHandler<npy_uint32>::validate(FFIType type_code) {
        if (type_code != FFIType::NUMPY_UnsignedInt32 && type_code != FFIType::UnsignedInt) {
            throw std::runtime_error(
                    "type 'unsigned int'/'npy_uint32' misaligned with type code; expected 'UnsignedInt'/'NUMPY_UnsignedInt32'"
                    );
        };
    }
    template <>
    void  FFITypeHandler<npy_uint64>::validate(FFIType type_code) {
        if (type_code != FFIType::NUMPY_UnsignedInt64 && type_code != FFIType::UnsignedLong) {
            throw std::runtime_error("type 'unsigned long'/'npy_uint64' misaligned with type code; expected 'UnsignedLong'/'NUMPY_UnsignedInt64'");
        };
    }
//        template <>
//        void  FFITypeHandler<npy_float16>::validate(FFIType type_code) {
//            if (type_code != FFIType::NUMPY_Float16) {
//                throw std::runtime_error("type 'npy_float16' misaligned with type code; expected 'NUMPY_Float16'");
//            };
//        }
    template <>
    void  FFITypeHandler<npy_float32>::validate(FFIType type_code) {
        if (type_code != FFIType::NUMPY_Float32 && type_code != FFIType::Float) {
            throw std::runtime_error("type 'float'/'npy_float32' misaligned with type code; expected 'Float'/'NUMPY_Float32'");
        };
    }
    template <>
    void  FFITypeHandler<npy_float64>::validate(FFIType type_code) {
        if (type_code != FFIType::NUMPY_Float64 && type_code != FFIType::Double) {
            throw std::runtime_error("type 'double'/'npy_float64' misaligned with type code; expected 'NUMPY_Float64'");
        };
    }
    template <>
    void  FFITypeHandler<npy_float128>::validate(FFIType type_code) {
        if (type_code != FFIType::NUMPY_Float128) {
            throw std::runtime_error("type 'npy_float128' misaligned with type code; expected 'NUMPY_Float128'");
        };
    }

//        template <>
//        void  FFITypeHandler<unsigned char>::validate(FFIType type_code) {
//            if (type_code != FFIType::UnsignedChar) {
//                throw std::runtime_error("type 'unsigned char'/'npy_bool' misaligned with type code; expected 'UnsignedChar'/'NUMPY_Bool'");
//            };
//        }
//        template <>
//        void  FFITypeHandler<short>::validate(FFIType type_code) {
//            if (type_code != FFIType::Short && type_code != FFIType::NUMPY_Int16) {
//                throw std::runtime_error("type 'short'/'npy_int16' misaligned with type code; expected 'Short'/'NUMPY_Int16'");
//            };
//        }
//        template <>
//        void  FFITypeHandler<unsigned short >::validate(FFIType type_code) {
//            if (type_code != FFIType::UnsignedShort) {
//                throw std::runtime_error("type 'unsigned short ' misaligned with type code; expected 'UnsignedShort'");
//            };
//        }
//        template <>
//        void  FFITypeHandler<int >::validate(FFIType type_code) {
//            if (type_code != FFIType::Int) {
//                throw std::runtime_error("type 'int ' misaligned with type code; expected 'Int'");
//            };
//        }
//        template <>
//        void  FFITypeHandler<unsigned int >::validate(FFIType type_code) {
//            if (type_code != FFIType::UnsignedInt) {
//                throw std::runtime_error("type 'unsigned int ' misaligned with type code; expected 'UnsignedInt'");
//            };
//        }
//        template <>
//        void  FFITypeHandler<long >::validate(FFIType type_code) {
//            if (type_code != FFIType::Long && type_code != FFIType::PySizeT) {
//                throw std::runtime_error("type 'long' a.k.a `Py_Ssize_t` misaligned with type code; expected 'Long'/'PySizeT'");
//            };
//        }
//        template <>
//        void  FFITypeHandler<unsigned long >::validate(FFIType type_code) {
//            if (type_code != FFIType::UnsignedLong) {
//                throw std::runtime_error("type 'unsigned long ' misaligned with type code; expected 'UnsignedLong'");
//            };
//        }
    template <>
    void  FFITypeHandler<long long >::validate(FFIType type_code) {
        if (type_code != FFIType::LongLong) {
            throw std::runtime_error("type 'long long ' misaligned with type code; expected 'LongLong'");
        };
    }
    template <>
    void  FFITypeHandler<unsigned long long >::validate(FFIType type_code) {
        if (type_code != FFIType::UnsignedLongLong) {
            throw std::runtime_error("type 'unsigned long long ' misaligned with type code; expected 'UnsignedLongLong'");
        };
    }
//        template <>
//        void  FFITypeHandler<Py_ssize_t >::validate(FFIType type_code) {
//            if (type_code != FFIType::PySizeT) {
//                throw std::runtime_error("type 'Py_ssize_t ' misaligned with type code; expected 'PySizeT'");
//            };
//        }
//        template <>
//        void  FFITypeHandler<float >::validate(FFIType type_code) {
//            if (type_code != FFIType::Float) {
//                throw std::runtime_error("type 'float' misaligned with type code; expected 'Float'");
//            };
//        }
//        template <>
//        void  FFITypeHandler<double >::validate(FFIType type_code) {
//            if (type_code != FFIType::Double) {
//                throw std::runtime_error("type 'double ' misaligned with type code; expected 'Double'");
//            };
//        }
    template <>
    void  FFITypeHandler<bool >::validate(FFIType type_code) {
        if (type_code != FFIType::Bool) {
            throw std::runtime_error("type 'bool ' misaligned with type code; expected 'Bool'");
        };
    }
    template <>
    void  FFITypeHandler<std::string>::validate(FFIType type_code) {
        if (type_code != FFIType::String) {
            throw std::runtime_error("type 'std::string ' misaligned with type code; expected 'String'");
        };
    }

//        template <typename T>
//        T FFITypeHandler<T>::cast(FFIType type_code, void *data) {
//            validate(type_code);
//            return *(T*)data;
//        }
//        template <typename T>
//        T* FFITypeHandler<T*>::cast(FFIType type_code, void *data) {
//            validate(type_code);
//            return (T*)data;
//        }

    std::shared_ptr<void> pyobj_to_voidptr(FFIType type_char, PyObject* py_obj) {
        switch(type_char) {
            case (FFIType::PyObject): {
                return std::shared_ptr<void>(
                        get_python_attr<PyObject *>(py_obj, "arg_data"),
                        [](PyObject*) {}
                );
            }
            case FFIType::UnsignedChar: {
                return std::make_shared<unsigned char>(
                        get_python_attr<unsigned char>(py_obj, "arg_data")
                );
            }
            case FFIType::Short: {
                return std::make_shared<short>(
                        get_python_attr<short>(py_obj, "arg_data")
                );
            }
            case FFIType::UnsignedShort: {
                return std::make_shared<unsigned short>(
                        get_python_attr<unsigned short>(py_obj, "arg_data")
                );
            }
            case FFIType::Int: {
                return std::make_shared<int>(
                        get_python_attr<int>(py_obj, "arg_data")
                );
            }
            case FFIType::UnsignedInt: {
                return std::make_shared<unsigned int>(
                        get_python_attr<unsigned int>(py_obj, "arg_data")
                );
            }
            case FFIType::Long: {
                return std::make_shared<long>(
                        get_python_attr<long>(py_obj, "arg_data")
                );
            }
            case FFIType::UnsignedLong: {
                return std::make_shared<unsigned long>(
                        get_python_attr<unsigned long>(py_obj, "arg_data")
                );
            }
            case FFIType::LongLong: {
                return std::make_shared<long long>(
                        get_python_attr<long long>(py_obj, "arg_data")
                );
            }
            case FFIType::UnsignedLongLong: {
                return std::make_shared<unsigned long long>(
                        get_python_attr<unsigned long long >(py_obj, "arg_data")
                );
            }
            case FFIType::PySizeT:{
                return std::make_shared<Py_ssize_t>(
                        get_python_attr<Py_ssize_t>(py_obj, "arg_data")
                );
            }
            case FFIType::Float:{
                return std::make_shared<float>(
                        get_python_attr<float>(py_obj, "arg_data")
                );
            }
            case FFIType::Double:{
                return std::make_shared<double>(
                        get_python_attr<double>(py_obj, "arg_data")
                );
            }
            case FFIType::Bool:{
                return std::make_shared<bool>(
                        get_python_attr<bool>(py_obj, "arg_data")
                );
            }
            case FFIType::String:{
                return std::make_shared<std::string>(
                        get_python_attr<std::string>(py_obj, "arg_data")
                );
            }

            case FFIType::NUMPY_Bool:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_bool>(py_obj, "arg_data"),
                        [](npy_bool *) {}
                );
            }
            case FFIType::NUMPY_Int8:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_int8>(py_obj, "arg_data"),
                        [](npy_int8 *) {}
                );
            }
            case FFIType::NUMPY_Int16:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_int16>(py_obj, "arg_data"),
                        [](npy_int16 *) {}
                );
            }
            case FFIType::NUMPY_Int32:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_int32>(py_obj, "arg_data"),
                        [](npy_int32 *) {}
                );
            }
            case FFIType::NUMPY_Int64:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_int64>(py_obj, "arg_data"),
                        [](npy_int64 *) {}
                );
            }
            case FFIType::NUMPY_UnsignedInt8:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_uint8>(py_obj, "arg_data"),
                        [](npy_uint8 *) {}
                );
            }
            case FFIType::NUMPY_UnsignedInt16:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_uint16>(py_obj, "arg_data"),
                        [](npy_uint16 *) {}
                );
            }
            case FFIType::NUMPY_UnsignedInt32:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_uint32>(py_obj, "arg_data"),
                        [](npy_uint32 *) {}
                );
            }
            case FFIType::NUMPY_UnsignedInt64:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_uint64>(py_obj, "arg_data"),
                        [](npy_uint64 *) {}
                );
            }

            case FFIType::NUMPY_Float16:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_float16>(py_obj, "arg_data"),
                        [](npy_float16 *) {}
                );
            }
            case FFIType::NUMPY_Float32:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_float32>(py_obj, "arg_data"),
                        [](npy_float32 *) {}
                );
            }
            case FFIType::NUMPY_Float64:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_float64>(py_obj, "arg_data"),
                        [](npy_float64 *) {}
                );
            }
            case FFIType::NUMPY_Float128:{
                return std::shared_ptr<void>(
                        get_python_attr_ptr<npy_float128>(py_obj, "arg_data"),
                        [](npy_float128 *) {}
                );
            }

            default:
                throw std::runtime_error("unhandled type specifier");
        }
    }

    void FFIParameter::init() {
        Py_XINCREF(py_obj);
        if (debug_print()) {
            auto garb = get_python_repr(py_obj);
            printf("Destructuring PyObject %s\n", garb.c_str());
        }
        if (debug_print()) printf("  > getting arg_type\n");
        auto type_char = get_python_attr<FFIType>(py_obj, "arg_type");
        if (debug_print()) printf("    > got %d\n", static_cast<int>(type_char));
        if (debug_print()) printf("  > getting arg_name\n");
        auto name = get_python_attr<std::string>(py_obj, "arg_name");
        if (debug_print()) printf("  > getting arg_shape\n");
        auto shape = get_python_attr_iterable<size_t>(py_obj, "arg_shape");
        if (debug_print()) printf("  > getting arg_val\n");
        auto val_obj = get_python_attr<PyObject*>(py_obj, "arg_value");
        if (debug_print()) printf("  converting to voidptr...\n");

        param_data = pyobj_to_voidptr(type_char, val_obj);
        Py_XDECREF(val_obj); // annoying...

        if (debug_print()) printf("  constructing FFIArgument...\n");

        arg_spec = FFIArgument(name, type_char, shape);

    }

     PyObject* FFIParameter::as_python() {
        switch(type()) {
            case (FFIType::PyObject): { return (PyObject*) param_data.get(); }
            case FFIType::UnsignedChar: {
                auto shp=shape();
                return FFITypeHandler<unsigned char>().as_python(FFIType::UnsignedChar, param_data, shp); }
            case FFIType::Short: {
                auto shp=shape();
                return FFITypeHandler<short>().as_python(FFIType::Short, param_data, shp); }
            case FFIType::UnsignedShort: {
                auto shp=shape();
                return FFITypeHandler<unsigned short>().as_python(FFIType::UnsignedShort, param_data, shp); }
            case FFIType::Int: {
                auto shp=shape();
                return FFITypeHandler<int>().as_python(FFIType::Int, param_data, shp); }
            case FFIType::UnsignedInt: {
                auto shp=shape();
                return FFITypeHandler<unsigned int>().as_python(FFIType::UnsignedInt, param_data, shp); }
            case FFIType::Long: {
                auto shp=shape();
                return FFITypeHandler<long>().as_python(FFIType::Long, param_data, shp); }
            case FFIType::UnsignedLong: {
                auto shp=shape();
                return FFITypeHandler<unsigned long>().as_python(FFIType::UnsignedLong, param_data, shp); }
            case FFIType::LongLong: {
                auto shp=shape();
                return FFITypeHandler<long long>().as_python(FFIType::LongLong, param_data, shp); }
            case FFIType::UnsignedLongLong: {
                auto shp=shape();
                return FFITypeHandler<unsigned long long>().as_python(FFIType::UnsignedLongLong, param_data, shp); }
            case FFIType::PySizeT: {
                auto shp=shape();
                return FFITypeHandler<Py_ssize_t>().as_python(FFIType::PySizeT, param_data, shp); }
            case FFIType::Float: {
                auto shp=shape();
                return FFITypeHandler<float>().as_python(FFIType::Float, param_data, shp); }
            case FFIType::Double: {
                auto shp=shape();
                return FFITypeHandler<double>().as_python(FFIType::Double, param_data, shp); }
            case FFIType::Bool: {
                auto shp=shape();
                return FFITypeHandler<bool>().as_python(FFIType::Bool, param_data, shp); }
            case FFIType::NUMPY_Bool: {
                auto shp=shape();
                return FFITypeHandler<npy_bool*>().as_python(FFIType::NUMPY_Bool, param_data, shp); }
            case FFIType::NUMPY_Int8: {
                auto shp=shape();
                return FFITypeHandler<npy_int8*>().as_python(FFIType::NUMPY_Int8, param_data, shp); }
            case FFIType::NUMPY_Int16: {
                auto shp=shape();
                return FFITypeHandler<npy_int16*>().as_python(FFIType::NUMPY_Int16, param_data, shp); }
            case FFIType::NUMPY_Int32: {
                auto shp=shape();
                return FFITypeHandler<npy_int32*>().as_python(FFIType::NUMPY_Int32, param_data, shp); }
            case FFIType::NUMPY_Int64: {
                auto shp=shape();
                return FFITypeHandler<npy_int64*>().as_python(FFIType::NUMPY_Int64, param_data, shp); }
            case FFIType::NUMPY_UnsignedInt8: {
                auto shp=shape();
                return FFITypeHandler<npy_uint8*>().as_python(FFIType::NUMPY_UnsignedInt8, param_data, shp); }
            case FFIType::NUMPY_UnsignedInt16: {
                auto shp=shape();
                return FFITypeHandler<npy_uint16*>().as_python(FFIType::NUMPY_UnsignedInt16, param_data, shp); }
            case FFIType::NUMPY_UnsignedInt32: {
                auto shp=shape();
                return FFITypeHandler<npy_uint32*>().as_python(FFIType::NUMPY_UnsignedInt32, param_data, shp); }
            case FFIType::NUMPY_UnsignedInt64: {
                auto shp=shape();
                return FFITypeHandler<npy_uint64*>().as_python(FFIType::NUMPY_UnsignedInt64, param_data, shp); }
            case FFIType::NUMPY_Float16: {
                auto shp=shape();
                return FFITypeHandler<npy_float16*>().as_python(FFIType::NUMPY_Float16, param_data, shp); }
            case FFIType::NUMPY_Float32: {
                auto shp=shape();
                return FFITypeHandler<npy_float32*>().as_python(FFIType::NUMPY_Float32, param_data, shp); }
            case FFIType::NUMPY_Float64: {
                auto shp=shape();
                return FFITypeHandler<npy_float64*>().as_python(FFIType::NUMPY_Float64, param_data, shp); }
            case FFIType::NUMPY_Float128: {
                auto shp=shape();
                return FFITypeHandler<npy_float128*>().as_python(FFIType::NUMPY_Float128, param_data, shp); }
            default:
                throw std::runtime_error("unhandled type specifier");

        }

    }


//        template <typename T>
//        T FFIParameter::value() {
//            FFITypeHandler<T> handler;
//            return handler.cast(param_data);
//        }

    void FFIParameters::init() {
        params = get_python_attr_iterable<FFIParameter>(py_obj, "ffi_parameters");
    }

    size_t FFIParameters::param_index(std::string& param_name) {
        size_t i;
        for ( i=0; i < params.size(); i++) {
            auto p = params[i];
            if (p.name() == param_name) break;
        };
        if ( i > params.size()) throw std::runtime_error("parameter " + param_name + " not found");
        return i;
    }
    FFIParameter FFIParameters::get_parameter(std::string& param_name) {
        auto i = param_index(param_name);
        return params[i];
    }
    FFIParameter FFIParameters::get_parameter(const char* param_name) {
        std::string key = param_name;
        return get_parameter(key);
    }
    void FFIParameters::set_parameter(std::string& param_name, FFIParameter& param) {
        auto i = param_index(param_name);
        params[i] = param;
    }
    void FFIParameters::set_parameter(const char *param_name, FFIParameter &param) {
        std::string key = param_name;
        set_parameter(key, param);
    }

}