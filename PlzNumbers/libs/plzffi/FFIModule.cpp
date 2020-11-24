
#include "FFIModule.hpp"

namespace plzffi {

//        template <typename T>
//        T FFIMethod<T>::call(FFIParameters& params) {
//            return function_pointer(params);
//        }
//
//        template <typename T>
//        void FFIMethod<T>::type_check() {
//            FFITypeHandler<T> handler;
//            handler.validate(ret_type);
//        }
//
//        template <typename T>
//        void FFIModule::add_method(FFIMethod<T> method) {
//            method_names.push_back(method.name());
//            return_types.push_back(method.type());
//            method_pointers.push_back((void *)method);
//        }
//        template <typename T>
//        FFIMethod<T> FFIModule::get_method(std::string& method_name) {
//            for (auto i=0; i < method_names.size(); i++) {
//                if (method_names[i] == method_name) {
//                    return FFIModule::get_method_from_index<T>(i);
//                }
//            }
//            throw std::runtime_error("method " + method_name + "not found");
//        }
//
//        template <typename T>
//        FFIMethod<T> FFIModule::get_method_from_index(size_t i) {
//            FFITypeHandler<T> handler;
//            handler.validate(return_types[i]);
//            return *(FFIMethod<T>*) method_pointers[i];
//        }

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
                                                // but we'll need to be careful to avoid any memory leaks
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

        FFIModule ffi_from_capsule(PyObject *capsule) {
            FFIModule mod; // empty module
            return rynlib::python::from_python_capsule<FFIModule>(capsule, mod.ffi_module_attr().c_str());
        }

}
