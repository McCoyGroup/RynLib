
#ifndef RYNLIB_FFIMODULE_HPP
#define RYNLIB_FFIMODULE_HPP

#include "PyAllUp.hpp"
#include "FFIParameters.hpp"
#include <string>
#include <vector>

namespace rynlib {
    namespace PlzNumbers {

//        template <typename >
//        typedef T (*Func)(const FFIParameters&);

        template <typename T>
        class FFIMethod {
            std::string name;
            FFIType ret_type;
            T function_pointer;
        public:
            FFIMethod(
                    std::string& method_name,
                    FFIType return_type,
                    T function
                    ) : name(method_name), ret_type(return_type), function_pointer(function) { type_check(); };
            void type_check();
            T call(FFIParameters& params);
        };

        class FFIModule {
            // possibly memory leaky, but barely so & thus we won't worry too much until we _know_ it's an issue
            std::string name;
            std::string docstring;
            int size;
            std::string attr;
            std::vector<std::string> method_names;
            std::vector<FFIType> return_types;
            std::vector<void *> method_pointers; // pointers to FFI methods, but return types are ambiguous
        public:
            FFIModule(std::string& module_name, std::string& module_doc) : name(module_name), docstring(module_doc) {
                size = -1; // size of module per interpreter...for future use
                attr = "_FFIModule"; // attribute use when attaching to Python module
                return_types = {};
                method_names = {};
                method_pointers = {};
            }

            template <typename T>
            void add_method(FFIMethod<T> method);
            template <typename T>
            FFIMethod<T> get_method(std::string& method_name);
            template <typename T>
            FFIMethod<T> get_method_from_index(size_t i);

            // pieces necessary to hook into the python runtime
            PyObject *get_capsule();
            bool attach(PyObject* module);
            const char* doc();
            struct PyModuleDef get_def();
        };

        FFIModule from_capsule(PyObject* capsule) {
            return python::from_python_capsule<FFIModule>(capsule, "_FFIModule");
        }

    }
}

#endif //RYNLIB_FFIMODULE_HPP
