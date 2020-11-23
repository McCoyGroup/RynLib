
#include "FFIParameters.hpp"
#include <stdexcept>
#include <string>

namespace rynlib {

    using namespace python;
    namespace PlzNumbers {

        // defines a compiler map between FFIType and proper types
        template <typename T>
        FFIType infer_FFIType() {
            throw std::runtime_error("unhandled type specifier");
        }

        template <typename T>
        void FFITypeHandler<T>::validate(FFIType type_code) {
            throw std::runtime_error("unhandled type specifier");
        };
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

        template <typename T>
        T FFITypeHandler<T>::cast(FFIType type_code, void *data) {
            validate(type_code);
            return *(T*)data;
        }
        template <typename T>
        T* FFITypeHandler<T*>::cast(FFIType type_code, void *data) {
            validate(type_code);
            return (T*)data;
        }

        void FFIParameter::init() {
            type_char = get_python_attr<FFIType>(py_obj, "arg_type");
            param_key = get_python_attr<std::string>(py_obj, "arg_name");
            shape_vec = get_python_attr_iterable<size_t>(py_obj, "arg_shape");

            switch(type_char) {
                case (FFIType::PyObject): {
                    param_data = get_python_attr<PyObject *>(py_obj, "arg_data");
                    break;
                }
                case FFIType::UnsignedChar: {
                    auto dat = get_python_attr<unsigned char>(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::Short: {
                    auto dat = get_python_attr<short>(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::UnsignedShort: {
                    auto dat = get_python_attr<unsigned short >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::Int: {
                    auto dat = get_python_attr<int >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::UnsignedInt: {
                    auto dat = get_python_attr<unsigned int >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::Long: {
                    auto dat = get_python_attr<long >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::UnsignedLong: {
                    auto dat = get_python_attr<unsigned long >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::LongLong: {
                    auto dat = get_python_attr<long long >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::UnsignedLongLong: {
                    auto dat = get_python_attr<unsigned long long >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::PySizeT:{
                    auto dat = get_python_attr<Py_ssize_t >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::Float:{
                    auto dat = get_python_attr<float >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::Double:{
                    auto dat = get_python_attr<double >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::Bool:{
                    auto dat = get_python_attr<bool >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }
                case FFIType::String:{
                    auto dat = get_python_attr<std::string >(py_obj, "arg_data");
                    param_data = &dat;
                    break;
                }

                case FFIType::NUMPY_Bool:{
                    auto dat = get_python_attr_ptr<npy_bool>(py_obj, "arg_data");
                    param_data = dat; // already a ptr
                    break;
                }
                case FFIType::NUMPY_Int8:{
                    auto dat = get_python_attr_ptr<npy_int8>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_Int16:{
                    auto dat = get_python_attr_ptr<npy_int16>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_Int32:{
                    auto dat = get_python_attr_ptr<npy_int32>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_Int64:{
                    auto dat = get_python_attr_ptr<npy_int64>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_UnsignedInt8:{
                    auto dat = get_python_attr_ptr<npy_uint8>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_UnsignedInt16:{
                    auto dat = get_python_attr_ptr<npy_uint16>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_UnsignedInt32:{
                    auto dat = get_python_attr_ptr<npy_uint32>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_UnsignedInt64:{
                    auto dat = get_python_attr_ptr<npy_uint64>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }

                case FFIType::NUMPY_Float16:{
                    auto dat = get_python_attr_ptr<npy_float16>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_Float32:{
                    auto dat = get_python_attr_ptr<npy_float32>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_Float64:{
                    auto dat = get_python_attr_ptr<npy_float64>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }
                case FFIType::NUMPY_Float128:{
                    auto dat = get_python_attr_ptr<npy_float128>(py_obj, "arg_data");
                    param_data = dat;
                    break;
                }

                default:
                    throw std::runtime_error("unhandled type specifier");
            }
        }

        template <typename T>
        T FFIParameter::value() {
            FFITypeHandler<T> handler;
            return handler.cast(param_data);
        }

        void FFIParameters::init() {
            params = from_python_iterable<FFIParameter>(py_obj);
        }

        int FFIParameters::param_index(std::string& param_name) {
            int i;
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
        FFIParameter FFIParameters::set_parameter(std::string& param_name, FFIParameter& param) {
            auto i = param_index(param_name);
            params[i] = param;
        }

    }
}