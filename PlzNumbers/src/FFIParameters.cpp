
#include "FFIParameters.hpp"
#include <stdexcept>
#include <string>

namespace rynlib {

    using namespace python;
    namespace PlzNumbers {

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
        T FFIParameter::get_data() {
            throw std::runtime_error("unhandled type specifier");
        }
        template <>
        unsigned char FFIParameter::get_data<unsigned char>() {
            if (type_char != FFIType::UnsignedChar) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(unsigned char*)param_data; // cast to pointer then dereference
        }
        template <>
        short FFIParameter::get_data<short>() {
            if (type_char != FFIType::Short) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(short*)param_data; // cast to pointer then dereference
        }
        template <>
        unsigned short  FFIParameter::get_data<unsigned short >() {
            if (type_char != FFIType::UnsignedShort) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(unsigned short *)param_data; // cast to pointer then dereference
        }
        template <>
        int  FFIParameter::get_data<int >() {
            if (type_char != FFIType::Int) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(int *)param_data; // cast to pointer then dereference
        }
        template <>
        unsigned int  FFIParameter::get_data<unsigned int >() {
            if (type_char != FFIType::UnsignedInt) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(unsigned int *)param_data; // cast to pointer then dereference
        }
        template <>
        long  FFIParameter::get_data<long >() {
            if (type_char != FFIType::Long && type_char != FFIType::PySizeT) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(long *)param_data; // cast to pointer then dereference
        }
        template <>
        unsigned long  FFIParameter::get_data<unsigned long >() {
            if (type_char != FFIType::UnsignedLong) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(unsigned long *)param_data; // cast to pointer then dereference
        }
        template <>
        long long  FFIParameter::get_data<long long >() {
            if (type_char != FFIType::LongLong) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(long long *)param_data; // cast to pointer then dereference
        }
        template <>
        unsigned long long  FFIParameter::get_data<unsigned long long >() {
            if (type_char != FFIType::UnsignedLongLong) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(unsigned long long *)param_data; // cast to pointer then dereference
        }
//        template <>
//        Py_ssize_t  FFIParameter::get_data<Py_ssize_t >() {
//            if (type_char != FFIType::PySizeT) {
//                throw std::runtime_error("requested parameter data type doesn't match held data");
//            }
//            return *(Py_ssize_t *)param_data; // cast to pointer then dereference
//        }
        template <>
        float  FFIParameter::get_data<float >() {
            if (type_char != FFIType::Float) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(float *)param_data; // cast to pointer then dereference
        }
        template <>
        double  FFIParameter::get_data<double >() {
            if (type_char != FFIType::Double) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(double *)param_data; // cast to pointer then dereference
        }
        template <>
        bool  FFIParameter::get_data<bool >() {
            if (type_char != FFIType::Bool) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(bool *)param_data; // cast to pointer then dereference
        }
        template <>
        std::string  FFIParameter::get_data<std::string >() {
            if (type_char != FFIType::String) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return *(std::string *)param_data; // cast to pointer then dereference
        }
        template <>
        npy_bool* FFIParameter::get_data<npy_bool*>() {
            if (type_char != FFIType::NUMPY_Bool && type_char != FFIType::NUMPY_UnsignedInt8) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_bool*)param_data; // cast to pointer
        }
        template <>
        npy_int8* FFIParameter::get_data<npy_int8*>() {
            if (type_char != FFIType::NUMPY_Int8) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_int8*)param_data; // cast to pointer
        }
        template <>
        npy_int16* FFIParameter::get_data<npy_int16*>() {
            if (type_char != FFIType::NUMPY_Int16) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_int16*)param_data; // cast to pointer
        }
        template <>
        npy_int32* FFIParameter::get_data<npy_int32*>() {
            if (type_char != FFIType::NUMPY_Int32) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_int32*)param_data; // cast to pointer
        }
        template <>
        npy_int64* FFIParameter::get_data<npy_int64*>() {
            if (type_char != FFIType::NUMPY_Int64) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_int64*)param_data; // cast to pointer
        }
//        template <>
//        npy_uint8* FFIParameter::get_data<npy_uint8*>() {
//            if (type_char != FFIType::NUMPY_UnsignedInt8) {
//                throw std::runtime_error("requested parameter data type doesn't match held data");
//            }
//            return (npy_uint8*)param_data; // cast to pointer
//        }
        template <>
        npy_uint16* FFIParameter::get_data<npy_uint16*>() {
            if (type_char != FFIType::NUMPY_UnsignedInt16 && type_char != FFIType::NUMPY_Float16) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_uint16*)param_data; // cast to pointer
        }
        template <>
        npy_uint32* FFIParameter::get_data<npy_uint32*>() {
            if (type_char != FFIType::NUMPY_UnsignedInt32) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_uint32*)param_data; // cast to pointer
        }
        template <>
        npy_uint64* FFIParameter::get_data<npy_uint64*>() {
            if (type_char != FFIType::NUMPY_UnsignedInt64) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_uint64*)param_data; // cast to pointer
        }
//        template <>
//        npy_float16* FFIParameter::get_data<npy_float16*>() {
//            if (type_char != FFIType::NUMPY_Float16) {
//                throw std::runtime_error("requested parameter data type doesn't match held data");
//            }
//            return (npy_float16*)param_data; // cast to pointer
//        }
        template <>
        npy_float32* FFIParameter::get_data<npy_float32*>() {
            if (type_char != FFIType::NUMPY_Float32) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_float32*)param_data; // cast to pointer
        }
        template <>
        npy_float64* FFIParameter::get_data<npy_float64*>() {
            if (type_char != FFIType::NUMPY_Float64) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_float64*)param_data; // cast to pointer
        }
        template <>
        npy_float128* FFIParameter::get_data<npy_float128*>() {
            if (type_char != FFIType::NUMPY_Float128) {
                throw std::runtime_error("requested parameter data type doesn't match held data");
            }
            return (npy_float128*)param_data; // cast to pointer
        }

    }
}