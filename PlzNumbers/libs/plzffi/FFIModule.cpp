
#include "FFIModule.hpp"

namespace plzffi {

    using rynlib::python::py_printf;

    void FFIModule::init() {
        capsule_name = name + "." + attr;
    }

    PyObject *FFIModule::get_capsule() {
//            auto full_name = ffi_module_attr();
//            printf("wat %s\n", capsule_name.c_str());
        auto cap = PyCapsule_New((void *) this, capsule_name.c_str(), NULL); // do I need a destructor?
        return Py_BuildValue(
                "(NN)",
                get_py_name(),
                cap
        );
    }

    PyObject *FFIModule::get_py_name() {
        return rynlib::python::as_python<std::string>(name);
    }

    bool FFIModule::attach(PyObject *module) {
        PyObject *capsule = get_capsule();
        if (capsule == NULL) return false;
        bool i_did_good = (PyModule_AddObject(module, attr.c_str(), capsule) == 0);
        if (!i_did_good) {
            Py_XDECREF(capsule);
            Py_DECREF(module);
        } else {
            PyObject *pyname = get_py_name();
            i_did_good = (PyModule_AddObject(module, "name", pyname) == 0);
            if (!i_did_good) {
                Py_XDECREF(capsule);
                Py_XDECREF(pyname);
                Py_DECREF(module);
            }
        }

        return i_did_good;
    }

    const char *FFIModule::doc() {
        return docstring.c_str();
    }

    struct PyModuleDef FFIModule::get_def() {
        // once I have them, I should hook into python methods to return, e.g. the method names and return types
        // inside the module
        auto *methods = new PyMethodDef[5]; // I think Python manages this memory if def() only gets called once
        // but we'll need to be careful to avoid any memory leaks
        methods[0] = {"get_signature", _pycall_python_signature, METH_VARARGS, "gets the signature for an FFI module"};
        methods[1] = {"get_name", _pycall_module_name, METH_VARARGS, "gets the module name for an FFI module"};
        methods[2] = {"call_method", _pycall_evaluate_method, METH_VARARGS, "calls a method from an FFI module"};
        methods[3] = {"call_method_threaded", _pycall_evaluate_method_threaded, METH_VARARGS, "calls a method from an FFI module using a threading strategey"};
        methods[4] = {NULL, NULL, 0, NULL};
        return {
                PyModuleDef_HEAD_INIT,
                name.c_str(),   /* name of module */
                doc(), /* module documentation, may be NULL */
                size,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
                methods
        };
    }

    PyObject *FFIModule::python_signature() {

        std::vector<PyObject *> py_sigs(method_data.size(), NULL);
        for (size_t i = 0; i < method_data.size(); i++) {

            auto args = method_data[i].args;
            if (debug_print()) py_printf(" > constructing signature for %s\n",
                                      method_data[i].name.c_str());
//                    printf("....wat %lu\n", args.size());
            std::vector<PyObject *> subargs(args.size(), NULL);
            for (size_t j = 0; j < args.size(); j++) {
                subargs[j] = args[j].as_tuple();
            }

            py_sigs[i] = Py_BuildValue(
                    "(NNNN)",
                    rynlib::python::as_python<std::string>(method_data[i].name),
                    rynlib::python::as_python_tuple<PyObject *>(subargs),
                    rynlib::python::as_python<int>(static_cast<int>(method_data[i].ret_type)), // to be python portable
                    rynlib::python::as_python<bool>(method_data[i].vectorized)
            );
        }

        return Py_BuildValue(
                "(NN)",
                rynlib::python::as_python<std::string>(name),
                rynlib::python::as_python_tuple<PyObject *>(py_sigs)
        );

    }

    FFIModule ffi_from_capsule(PyObject *captup) {
//        set_debug_print(true); // temporary debug hack
        if (!PyTuple_Check(captup)) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "FFIModule spec. expected to be a tuple looking like (name, capsule)"
            );
            throw std::runtime_error("bad tuple shiz");
        }

        if (debug_print()) py_printf("Got FFIModule spec \"%s\"\n", rynlib::python::get_python_repr(captup).c_str());
        auto name_obj = PyTuple_GetItem(captup, 0);
        if (name_obj == NULL) throw std::runtime_error("bad tuple indexing");
        if (debug_print())
            py_printf("Pulling FFIModule for module \"%s\"\n", rynlib::python::get_python_repr(name_obj).c_str());
        auto cap_obj = PyTuple_GetItem(captup, 1);
        if (cap_obj == NULL) throw std::runtime_error("bad tuple indexing");
        if (debug_print())
            py_printf("  extracting from capsule \"%s\"\n", rynlib::python::get_python_repr(cap_obj).c_str());
        std::string name = rynlib::python::from_python<std::string>(name_obj);
        std::string doc;
        FFIModule mod(name, doc); // empty module
        if (debug_print()) py_printf("  pulling pointer with name \"%s\"\n", mod.ffi_module_attr().c_str());
        return rynlib::python::from_python_capsule<FFIModule>(cap_obj, mod.ffi_module_attr().c_str());
    }

    size_t FFIModule::get_method_index(std::string &method_name) {
        for (size_t i = 0; i < method_data.size(); i++) {
            if (method_data[i].name == method_name) { return i; }
        }
        throw std::runtime_error("method " + method_name + "not found");
    }

    FFIMethodData FFIModule::get_method_data(std::string &method_name) {
        for (auto data : method_data) {
            if (data.name == method_name) {
//                py_printf("Method %s is the %lu-th method in %s\n", method_name.c_str(), i, name.c_str());
                return data;
            }
        }
        throw std::runtime_error("method " + method_name + " not found");
    }

    PyObject* FFIModule::py_call_method(PyObject *method_name, PyObject *params) {

        if (debug_print()) py_printf("Calling from python ");
        auto mname = rynlib::python::from_python<std::string>(method_name);
        if (debug_print()) py_printf(" into method %s\n", mname.c_str());
        auto meth_idx = get_method_index(mname);
        auto argtype = method_data[meth_idx].ret_type;

        if (debug_print()) py_printf(" > loading parameters...\n");
        auto args = FFIParameters(params);

        if (debug_print()) py_printf(" > calling on parameters...\n");
        return ffi_call_method(
                argtype,
                *this,
                mname,
                args
                );
    }

    PyObject *FFIModule::py_call_method_threaded(PyObject *method_name,
                                                 PyObject *params,
                                                 PyObject *looped_var,
                                                 PyObject *threading_mode
                                                 ) {

        auto mname = rynlib::python::from_python<std::string>(method_name);
        auto meth_idx = get_method_index(mname);
        auto argtype = method_data[meth_idx].ret_type;
        auto args = FFIParameters(params);

        auto varname = rynlib::python::from_python<std::string>(looped_var);
        auto mode = rynlib::python::from_python<std::string>(threading_mode);
        auto thread_var = args.get_parameter(varname);
        auto ttype = thread_var.type();

        return ffi_call_method_threaded(
                argtype,
                ttype,
                *this,
                mname,
                varname, mode,
                args
        );

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

    PyObject *_pycall_evaluate_method(PyObject *self, PyObject *args) {
        PyObject *cap, *method_name, *params, *looped_var, *threading_mode;
        auto parsed = PyArg_ParseTuple(args, "OOO",
                                       &cap,
                                       &method_name,
                                       &params
                                       );
        if (!parsed) { return NULL; }

//        set_debug_print(true);
//        rynlib::python::pyadeeb.set_debug_print(true);

        try {
            auto obj = ffi_from_capsule(cap);
            return obj.py_call_method(method_name, params);
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

    PyObject *_pycall_evaluate_method_threaded(PyObject *self, PyObject *args) {
        PyObject *cap, *method_name, *params, *looped_var, *threading_mode;
        auto parsed = PyArg_ParseTuple(args, "OOOOO", &cap, &method_name, &params, &looped_var, &threading_mode);
        if (!parsed) { return NULL; }

        try {
            auto obj = ffi_from_capsule(cap);
            return obj.py_call_method_threaded(method_name, params, looped_var, threading_mode);
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
