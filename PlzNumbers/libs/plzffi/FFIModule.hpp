
#ifndef RYNLIB_FFIMODULE_HPP
#define RYNLIB_FFIMODULE_HPP

#include "PyAllUp.hpp"
#include "FFIParameters.hpp"
#include <string>
#include <vector>
#include <stdexcept>

namespace plzffi {

//        template <typename >
//        typedef T (*Func)(const FFIParameters&);

        template <typename T>
        class FFIMethod {
            std::string name;
            std::vector<FFIArgument> args;
            FFIType ret_type;
            T (*function_pointer)(FFIParameters&);
        public:
            FFIMethod(
                    std::string& method_name,
                    std::vector<FFIArgument>& arg,
                    FFIType return_type,
                    T (*function)(FFIParameters&)
                    ) : name(method_name), args(arg), ret_type(return_type), function_pointer(function) { type_check(); };
            FFIMethod(
                    const char* method_name,
                    std::vector<FFIArgument> arg,
                    FFIType return_type,
                    T (*function)(FFIParameters&)
            ) : name(method_name), args(arg), ret_type(return_type), function_pointer(function) { type_check(); };
            void type_check();
            T call(FFIParameters& params);

            std::string method_name() {return name;}
            std::vector<FFIArgument> method_arguments() {return args;}
            FFIType return_type() {return ret_type;}

//            PyObject * python_signature() {
//                std::vector<PyObject*> py_args(args.size(), NULL);
//                for (size_t i=0; i < args.size(); i++) {
//                    py_args[i] = args[i].as_tuple();
//                }
//                return Py_BuildValue(
//                        "(NNN)",
//                        rynlib::python::as_python<std::string>(name),
//                        rynlib::python::as_python_tuple<PyObject *>(py_args),
//                        rynlib::python::as_python<int>(static_cast<int>(ret_type))
//                        );
//            }

        };

        enum class FFIThreadingMode {
            OpenMP,
            TBB,
            SERIAL
        };

        template <typename T>
        class FFIThreader {
            FFIMethod<T> method;
            FFIThreadingMode mode;
        public:
            FFIThreader(FFIMethod<T>& method, FFIThreadingMode mode) : method(method), mode(mode) {}
            FFIThreader(FFIMethod<T>& method, std::string& mode_name) : method(method) {
                if (mode_name == "OpenMP") {
                    mode = FFIThreadingMode::OpenMP;
                } else if (mode_name == "TBB") {
                    mode = FFIThreadingMode::TBB;
                } else if (mode_name == "serial") {
                    mode = FFIThreadingMode::SERIAL;
                } else {
                    throw std::runtime_error("FFIThreader: unknown threading method");
                }
            }

            std::vector<T>
            call(FFIParameters& params, std::string& var) {
                auto threaded_param = params.get_parameter(var);

            }
        };

        class FFIModule {
            // possibly memory leaky, but barely so & thus we won't worry too much until we _know_ it's an issue
            std::string name;
            std::string docstring;
            int size = -1; // size of module per interpreter...for future use
            std::string attr = "_FFIModule"; // attribute use when attaching to Python module
            std::string capsule_name;
            std::vector<std::string> method_names;
            std::vector<std::vector<FFIArgument> > method_args;
            std::vector<FFIType> return_types;
            std::vector<void *> method_pointers; // pointers to FFI methods, but return types are ambiguous
        public:
            FFIModule() = default;
            FFIModule(std::string &module_name, std::string &module_doc) :
                    name(module_name),
                    docstring(module_doc) { init(); }
//                    return_types({}),
//                    method_names({}),
//                    method_pointers({}) {};
            FFIModule(const char* module_name, const char* module_doc) :
                    name(module_name),
                    docstring(module_doc) { init(); }

            void init();

            template <typename T>
            void add_method(FFIMethod<T>& method);
            template<typename T>
            void add(const char* method_name,
                            std::vector<FFIArgument> arg,
                            FFIType return_type,
                            T (*function)(FFIParameters&));
            template <typename T>
            FFIMethod<T> get_method(std::string& method_name);
            template <typename T>
            FFIMethod<T> get_method_from_index(size_t i);

            // pieces necessary to hook into the python runtime
            PyObject* get_py_name();
            PyObject *get_capsule();
            bool attach(PyObject* module);
            const char* doc();
            struct PyModuleDef get_def();
            std::string ffi_module_attr() { return capsule_name; };

            template<typename T>
            T call_method(std::string& method_name, FFIParameters& params) {
                return get_method<T>(method_name).call(params);
            }
            template<typename T>
            std::vector<T> call_method_threaded(std::string& method_name, FFIParameters& params, std::string& threaded_var, std::string& mode) {
                return FFIThreader<T>(get_method<T>(method_name), mode).call(params, threaded_var);
            }

            PyObject *python_signature();
            PyObject *py_call_method(PyObject* method_name, PyObject* params);
            PyObject *py_call_method_threaded(PyObject* method_name, PyObject* params, PyObject* looped_var, PyObject* threading_mode);

        };

        FFIModule ffi_from_capsule(PyObject* capsule);

        //region Template Fuckery
        template <typename T>
        T FFIMethod<T>::call(FFIParameters& params) {
            return function_pointer(params);
        }

        template <typename T>
        void FFIMethod<T>::type_check() {
            FFITypeHandler<T> handler;
            handler.validate(ret_type);
        }

        template <typename T>
        void FFIModule::add_method(FFIMethod<T>& method) {
            method_names.push_back(method.method_name());
            method_args.push_back(method.method_arguments());
            return_types.push_back(method.return_type());
            method_pointers.push_back((void *) &method);
        }

        template<typename T>
        void FFIModule::add(
                const char *method_name,
                std::vector<FFIArgument> arg,
                FFIType return_type,
                T (*function)(FFIParameters &)) {
            // need to introduce destructor to FFIModule to clean up all of these methods once we go out of scope
            auto meth = new FFIMethod<T>(method_name, arg, return_type, function);
            add_method(*meth);
        }

        template <typename T>
        FFIMethod<T> FFIModule::get_method(std::string& method_name) {
//            printf("Uh...?\n");
            for (size_t i=0; i < method_names.size(); i++) {
                if (method_names[i] == method_name) {
                    printf("Method %s is the %lu-th method in %s\n", method_name.c_str(), i, name.c_str());
                    return FFIModule::get_method_from_index<T>(i);
                }
            }
            throw std::runtime_error("method " + method_name + "not found");
        }

        template <typename T>
        FFIMethod<T> FFIModule::get_method_from_index(size_t i) {

//            printf("Making a thing...?\n");
            FFITypeHandler<T> handler;
            handler.validate(return_types[i]);
//            printf("Validated return type?\n");
            auto methodptr = static_cast<FFIMethod<T>*>(method_pointers[i]);
//            printf("Constructed pointer?\n");

            if (methodptr == NULL) {
                std::string err = "Bad pointer for method '%s'" + method_names[i];
                throw std::runtime_error(err.c_str());
            }


//            printf("Got name %s\n", methodptr->method_name().c_str());

            auto method = *methodptr;
//            printf("Dereferenced pointer?\n");

            return method;
        }


        //endregion

    PyObject * _pycall_python_signature(PyObject* self, PyObject* args);
    PyObject * _pycall_module_name(PyObject* self, PyObject* args);
    PyObject * _pycall_evaluate_method(PyObject* self, PyObject* args);
    PyObject * _pycall_evaluate_method_threaded(PyObject* self, PyObject* args);

}

#endif //RYNLIB_FFIMODULE_HPP
