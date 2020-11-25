
#include "FFIModule.hpp"

namespace plzffi {

        void FFIModule::init() {
            capsule_name = name + "." + attr;
        }

        PyObject* FFIModule::get_capsule() {
//            auto full_name = ffi_module_attr();
//            printf("wat %s\n", capsule_name.c_str());
            auto cap = PyCapsule_New((void *)this, capsule_name.c_str(), NULL); // do I need a destructor?
            return Py_BuildValue(
                    "(NN)",
                    get_py_name(),
                    cap
                    );
        }

        PyObject* FFIModule::get_py_name() {
            return rynlib::python::as_python<std::string>(name);
        }

        bool FFIModule::attach(PyObject* module) {
            PyObject* capsule = get_capsule();
            if (capsule == NULL) return false;
            bool i_did_good = (PyModule_AddObject(module, attr.c_str(), capsule) == 0);
            if (!i_did_good) {
                Py_XDECREF(capsule);
                Py_DECREF(module);
            } else {
                PyObject* pyname = get_py_name();
                i_did_good = (PyModule_AddObject(module, "name", pyname) == 0);
                if (!i_did_good) {
                    Py_XDECREF(capsule);
                    Py_XDECREF(pyname);
                    Py_DECREF(module);
                }
            }

            return i_did_good;
        }

        const char* FFIModule::doc() {
            return docstring.c_str();
        }

        struct PyModuleDef FFIModule::get_def() {
            // once I have them, I should hook into python methods to return, e.g. the method names and return types
            // inside the module
            auto* methods = new PyMethodDef[3]; // I think Python manages this memory if def() only gets called once
                                                // but we'll need to be careful to avoid any memory leaks
            methods[0] = {"get_signature", _pycall_python_signature, METH_VARARGS, "gets the signature for an FFI module"},
            methods[1] = {"get_name", _pycall_module_name, METH_VARARGS, "gets the module name for an FFI module"},
            methods[2] = {NULL, NULL, 0, NULL};
            return  {
                    PyModuleDef_HEAD_INIT,
                    name.c_str(),   /* name of module */
                    doc(), /* module documentation, may be NULL */
                    size,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
                    methods
            };
        }

        PyObject * FFIModule::python_signature() {

                std::vector<PyObject*> py_sigs(method_args.size(), NULL);
                for (size_t i=0; i < method_args.size(); i++) {

                    auto args = method_args[i];
//                    printf("....wat %lu\n", args.size());
                    std::vector<PyObject*> subargs(args.size(), NULL);
                    for (size_t j=0; j < args.size(); j++) {
                        subargs[j] = args[j].as_tuple();
                    }

                    py_sigs[i] = Py_BuildValue(
                            "(NNN)",
                            rynlib::python::as_python<std::string>(method_names[i]),
                            rynlib::python::as_python_tuple<PyObject *>(subargs),
                            rynlib::python::as_python<int>(static_cast<int>(return_types[i])) // to be python portable
                    );
                }

                return Py_BuildValue(
                        "(NN)",
                        rynlib::python::as_python<std::string>(name),
                        rynlib::python::as_python_tuple<PyObject *>(py_sigs)
                );

        }

        FFIModule ffi_from_capsule(PyObject *captup) {
            auto name_obj = PyTuple_GetItem(captup, 0);
            if (name_obj == NULL) throw std::runtime_error("bad tuple indexing");
            auto cap_obj = PyTuple_GetItem(captup, 1);
            if (cap_obj == NULL) throw std::runtime_error("bad tuple indexing");
            std::string name = rynlib::python::from_python<std::string>(name_obj);
            std::string doc;
            FFIModule mod(name, doc); // empty module
            return rynlib::python::from_python_capsule<FFIModule>(cap_obj, mod.ffi_module_attr().c_str());
        }

    PyObject *_pycall_python_signature(PyObject *self, PyObject *args) {

        PyObject *cap;
        auto parsed = PyArg_ParseTuple(args, "O", &cap);
        if (!parsed) { return NULL; }

        try {
            auto obj = ffi_from_capsule(cap);
//            printf("!!!!!!!?????\n");
            auto sig = obj.python_signature();

            return sig;
        } catch (std::exception &e) {
            if (!PyErr_Occurred()) {
                std::string msg = "in signature call: ";
                msg += e.what();
                PyErr_SetString(
                        PyExc_SystemError,
                        msg.c_str()
                );
            }
            return NULL;
        }

    }

    PyObject *_pycall_module_name(PyObject *self, PyObject *args) {

        PyObject *cap;
        auto parsed = PyArg_ParseTuple(args, "O", &cap);
        if (!parsed) { return NULL; }

        try {
            auto obj = ffi_from_capsule(cap);
//            printf(".....?????\n");
            auto name = obj.get_py_name();

            return name;
        } catch (std::exception &e) {
            if (!PyErr_Occurred()) {
                std::string msg = "in module_name call: ";
                msg += e.what();
                PyErr_SetString(
                        PyExc_SystemError,
                        msg.c_str()
                );
            }
            return NULL;
        }

    }

}
