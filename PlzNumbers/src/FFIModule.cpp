
#include "FFIModule.hpp"

namespace rynlib {
    namespace PlzNumbers {

        template <typename T>
        T FFIMethod<T>::call(FFIParameters& params) {
            return function_pointer(params);
        }

        template <typename T>
        void FFIMethod<T>::type_check() {
            throw std::runtime_error("unhandled dtype");
        }
        template <>
        void  FFIMethod<unsigned char>::type_check() {
            if (ret_type != FFIType::UnsignedChar) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<short>::type_check() {
            if (ret_type != FFIType::Short) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<unsigned short >::type_check() {
            if (ret_type != FFIType::UnsignedShort) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<int >::type_check() {
            if (ret_type != FFIType::Int) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<unsigned int >::type_check() {
            if (ret_type != FFIType::UnsignedInt) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<long >::type_check() {
            if (ret_type != FFIType::Long && ret_type != FFIType::PySizeT) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<unsigned long >::type_check() {
            if (ret_type != FFIType::UnsignedLong) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<long long >::type_check() {
            if (ret_type != FFIType::LongLong) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<unsigned long long >::type_check() {
            if (ret_type != FFIType::UnsignedLongLong) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
//        template <>
//        void  FFIMethod<Py_ssize_t >::type_check() {
//            if (ret_type != FFIType::PySizeT) {
//                throw std::runtime_error("specified method return type doesn't match template return type");
//            }
//        }
        template <>
        void  FFIMethod<float >::type_check() {
            if (ret_type != FFIType::Float) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<double >::type_check() {
            if (ret_type != FFIType::Double) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<bool >::type_check() {
            if (ret_type != FFIType::Bool) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<std::string >::type_check() {
            if (ret_type != FFIType::String) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }

        template <>
        void  FFIMethod<npy_bool*>::type_check() {
            if (ret_type != FFIType::NUMPY_Bool && ret_type != FFIType::NUMPY_UnsignedInt8) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<npy_int8*>::type_check() {
            if (ret_type != FFIType::NUMPY_Int8) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<npy_int16*>::type_check() {
            if (ret_type != FFIType::NUMPY_Int16) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<npy_int32*>::type_check() {
            if (ret_type != FFIType::NUMPY_Int32) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<npy_int64*>::type_check() {
            if (ret_type != FFIType::NUMPY_Int64) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
//        template <>
//        void  FFIMethod<npy_uint8*>::type_check() {
//            if (ret_type != FFIType::NUMPY_UnsignedInt8) {
//                throw std::runtime_error("specified method return type doesn't match template return type");
//            }
//        }
        template <>
        void  FFIMethod<npy_uint16*>::type_check() {
            if (ret_type != FFIType::NUMPY_UnsignedInt16 && ret_type != FFIType::NUMPY_Float16) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<npy_uint32*>::type_check() {
            if (ret_type != FFIType::NUMPY_UnsignedInt32) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<npy_uint64*>::type_check() {
            if (ret_type != FFIType::NUMPY_UnsignedInt64) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
//        template <>
//        void  FFIMethod<npy_float16*>::type_check() {
//            if (ret_type != FFIType::NUMPY_Float16) {
//                throw std::runtime_error("specified method return type doesn't match template return type");
//            }
//        }
        template <>
        void  FFIMethod<npy_float32*>::type_check() {
            if (ret_type != FFIType::NUMPY_Float32) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<npy_float64*>::type_check() {
            if (ret_type != FFIType::NUMPY_Float64) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }
        template <>
        void  FFIMethod<npy_float128*>::type_check() {
            if (ret_type != FFIType::NUMPY_Float128) {
                throw std::runtime_error("specified method return type doesn't match template return type");
            }
        }

        template <typename T>
        void FFIModule::add_method(FFIMethod<T> method) {
            method_names.push_back(method.name());
            return_types.push_back(method.type());
            method_pointers.push_back((void *)method);
        }
        template <typename T>
        FFIMethod<T> FFIModule::get_method(std::string& method_name) {
            for (auto i=0; i < method_names.size(); i++) {
                if (method_names[i] == method_name) {
                    return FFIModule::get_method_from_index<T>(i);
                }
            }

            throw std::runtime_error("method " + method_name + "not found");
        }

        template <typename T>
        FFIMethod<T> FFIModule::get_method_from_index(size_t i) {
            throw std::runtime_error("unhandled dtype");
        }
        template <>
        FFIMethod<unsigned char> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::UnsignedChar) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<unsigned char>*) method_pointers[i];
        }
        template <>
        FFIMethod<short> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::Short) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<short>*) method_pointers[i];
        }
        template <>
        FFIMethod<unsigned short > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::UnsignedShort) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<unsigned short >*) method_pointers[i];
        }
        template <>
        FFIMethod<int > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::Int) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<int >*) method_pointers[i];
        }
        template <>
        FFIMethod<unsigned int > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::UnsignedInt) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<unsigned int >*) method_pointers[i];
        }
        template <>
        FFIMethod<long > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::Long && return_types[i] != FFIType::PySizeT) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<long >*) method_pointers[i];
        }
        template <>
        FFIMethod<unsigned long > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::UnsignedLong) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<unsigned long >*) method_pointers[i];
        }
        template <>
        FFIMethod<long long > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::LongLong) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<long long >*) method_pointers[i];
        }
        template <>
        FFIMethod<unsigned long long > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::UnsignedLongLong) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<unsigned long long >*) method_pointers[i];
        }
//        template <>
//        FFIMethod<Py_ssize_t > FFIModule::get_method_from_index(size_t i) {
//            if (return_types[i] != FFIType::PySizeT) {
//                throw std::runtime_error("requested method return type doesn't match held return type");
//            }
//            return *(FFIMethod<Py_ssize_t >*) method_pointers[i];
//        }
        template <>
        FFIMethod<float > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::Float) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<float >*) method_pointers[i];
        }
        template <>
        FFIMethod<double > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::Double) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<double >*) method_pointers[i];
        }
        template <>
        FFIMethod<bool > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::Bool) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<bool >*) method_pointers[i];
        }
        template <>
        FFIMethod<std::string > FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::String) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<std::string >*) method_pointers[i];
        }

        template <>
        FFIMethod<npy_bool*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_Bool && return_types[i] != FFIType::NUMPY_UnsignedInt8 ) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_bool*>*) method_pointers[i];
        }
        template <>
        FFIMethod<npy_int8*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_Int8) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_int8*>*) method_pointers[i];
        }
        template <>
        FFIMethod<npy_int16*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_Int16) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_int16*>*) method_pointers[i];
        }
        template <>
        FFIMethod<npy_int32*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_Int32) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_int32*>*) method_pointers[i];
        }
        template <>
        FFIMethod<npy_int64*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_Int64) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_int64*>*) method_pointers[i];
        }
        template <>
//        FFIMethod<npy_uint8*> FFIModule::get_method_from_index(size_t i) {
//            if (return_types[i] != FFIType::NUMPY_UnsignedInt8) {
//                throw std::runtime_error("requested method return type doesn't match held return type");
//            }
//            return *(FFIMethod<npy_uint8*>*) method_pointers[i];
//        }
        template <>
        FFIMethod<npy_uint16*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_UnsignedInt16 && return_types[i] != FFIType::NUMPY_Float16) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_uint16*>*) method_pointers[i];
        }
        template <>
        FFIMethod<npy_uint32*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_UnsignedInt32) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_uint32*>*) method_pointers[i];
        }
        template <>
        FFIMethod<npy_uint64*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_UnsignedInt64) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_uint64*>*) method_pointers[i];
        }
//        template <>
//        FFIMethod<npy_float16*> FFIModule::get_method_from_index(size_t i) {
//            if (return_types[i] != FFIType::NUMPY_Float16) {
//                throw std::runtime_error("requested method return type doesn't match held return type");
//            }
//            return *(FFIMethod<npy_float16*>*) method_pointers[i];
//        }
        template <>
        FFIMethod<npy_float32*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_Float32) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_float32*>*) method_pointers[i];
        }
        template <>
        FFIMethod<npy_float64*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_Float64) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_float64*>*) method_pointers[i];
        }
        template <>
        FFIMethod<npy_float128*> FFIModule::get_method_from_index(size_t i) {
            if (return_types[i] != FFIType::NUMPY_Float128) {
                throw std::runtime_error("requested method return type doesn't match held return type");
            }
            return *(FFIMethod<npy_float128*>*) method_pointers[i];
        }

        PyObject* FFIModule::get_capsule() {
            auto full_name = name + "._FFIModule";
            return PyCapsule_New((void *)this, full_name.c_str(), NULL); // do I need a destructor?
        }

        bool FFIModule::attach(PyObject* module) {
            PyObject* capsule = get_capsule();
            if (capsule == NULL) return false;
            bool i_did_good = (PyModule_AddObject(module, attr.c_str(), capsule) == 0);
            if (!i_did_good) {
                Py_XDECREF(capsule);
                Py_DECREF(module);
            }

            return i_did_good;
        }

        const char* FFIModule::doc() {
            return docstring.c_str();
        }

        struct PyModuleDef FFIModule::get_def() {
            // once I have them, I should hook into python methods to return, e.g. the method names and return types
            // inside the module
            auto* methods = new PyMethodDef[1]; // I think Python manages this memory if def() only gets called once
            methods[0] = {NULL, NULL, 0, NULL};
            return  {
                    PyModuleDef_HEAD_INIT,
                    name.c_str(),   /* name of module */
                    doc(), /* module documentation, may be NULL */
                    size,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
                    methods
            };
        }



    }
}
