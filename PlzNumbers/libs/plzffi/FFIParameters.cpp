
#include "FFIParameters.hpp"
#include "PyAllUp.hpp"
#include <stdexcept>
#include <string>

namespace plzffi {

    using rynlib::python::py_printf;

    using namespace rynlib::python;

    bool DEBUG_PRINT=false;
    bool debug_print() {
        return DEBUG_PRINT;
    }
    void set_debug_print(bool db) {
        DEBUG_PRINT=db;
        pyadeeb.set_debug_print(db); // because of bad design choices I gotta do this multiple places...
    }

    // weirdness with constexpr... (https://stackoverflow.com/a/8016853/5720002)
    template <typename T, typename... Args>
    constexpr const FFIType FFIConversionManager<T, Args...>::pointer_types[];

    // defines a compiler map between FFIType and proper types
    PyObject * FFIArgument::as_tuple() {
        return Py_BuildValue("(NNN)",
                             rynlib::python::as_python<std::string>(param_key),
                             rynlib::python::as_python<FFIType>(type_char),
                             rynlib::python::as_python_tuple<size_t>(shape_vec)
        );
    }
    std::string FFIArgument::repr() {
        auto pp = as_tuple();
        auto repr = get_python_repr(pp);
        Py_XDECREF(pp);
        return repr;
    }

    void FFIParameter::init() {
        if (debug_print()) {
            auto garb = get_python_repr(py_obj);
            py_printf("Destructuring PyObject %s\n", garb.c_str());
        }
        if (debug_print()) py_printf("  > getting arg_type\n");
        auto type_char = get_python_attr<FFIType>(py_obj, "arg_type");
        if (debug_print()) py_printf("    > got %d\n", static_cast<int>(type_char));
        if (debug_print()) py_printf("  > getting arg_name\n");
        auto name = get_python_attr<std::string>(py_obj, "arg_name");
        if (debug_print()) py_printf("  > getting arg_shape\n");
        auto shape = get_python_attr_iterable<size_t>(py_obj, "arg_shape");
        if (debug_print()) py_printf("  > getting arg_val\n");
//        auto val_obj = get_python_attr<PyObject*>(py_obj, "arg_value");
//        if (debug_print()) py_printf("  converting to voidptr...\n");
        param_data = ffi_from_python_attr(type_char, py_obj, "arg_value", shape); // pulls arg_value by default...

        if (debug_print()) py_printf("  constructing FFIArgument...\n");

        arg_spec = FFIArgument(name, type_char, shape);

    }

     PyObject* FFIParameter::as_python() {
        auto shp = shape();
        return ffi_to_python(type(), param_data, shp);
    }

    std::string FFIParameter::repr() {
      auto pp = as_python();
      auto repr = get_python_repr(pp);
      Py_XDECREF(pp);
      return repr;
    }

    void FFIParameters::init() {
        params = get_python_attr_iterable<FFIParameter>(py_obj, "ffi_parameters");
    }

    size_t FFIParameters::param_index(std::string& param_name) {
        size_t i;
//        py_printf("  > looking through ");
//        py_printf("%lu poopies\n", params.size());
        for ( i=0; i < params.size(); i++) {
            auto p = params[i];
//            py_printf("  > this is not my mom (%lu) ", i);
//            py_printf("%s\n", p.name().c_str());
            if (p.name() == param_name) break;
        };
        if ( i == params.size()) throw std::runtime_error("parameter \"" + param_name + "\" not found");
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
        try {
            auto i = param_index(param_name);
            params[i] = param;
        } catch (std::exception& e) {
            params.push_back(param);
        }
    }
    void FFIParameters::set_parameter(const char *param_name, FFIParameter &param) {
        std::string key = param_name;
        set_parameter(key, param);
    }

    std::vector<size_t> FFIParameters::shape(std::string &key) {
        return get_parameter(key).shape();
    }
    std::vector<size_t> FFIParameters::shape(const char *key) {
        return get_parameter(key).shape();
    }
    FFIType FFIParameters::typecode(std::string &key) {
        return get_parameter(key).type();
    }
    FFIType FFIParameters::typecode(const char *key) {
        return get_parameter(key).type();
    }

}