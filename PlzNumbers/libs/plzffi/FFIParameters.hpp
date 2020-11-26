
#ifndef RYNLIB_FFIPARAMETERS_HPP
#define RYNLIB_FFIPARAMETERS_HPP

#include "PyAllUp.hpp"
#include <string>
#include <vector>
#include <memory>

namespace plzffi {

    bool debug_print();
    void set_debug_print(bool);

//    namespace PlzNumbers { class FFIParameter {
//        public: FFIParameter(PyObject*)
//    }; } //predeclare
//    template <>
//    inline PlzNumbers::FFIParameter python::from_python<PlzNumbers::FFIParameter>(PyObject *data) {
//        return PlzNumbers::FFIParameter(data);
//    }

    // Set up enum for type mapping
    // Must be synchronized with the types on the python side
    enum class FFIType {

        GENERIC = -1, // fallback for when things aren't really expected to have a type...

        PY_TYPES = 1000,
        UnsignedChar = PY_TYPES + 10,
        Short = PY_TYPES + 20,
        UnsignedShort = PY_TYPES + 21,
        Int = PY_TYPES + 30,
        UnsignedInt = PY_TYPES + 31,
        Long = PY_TYPES + 40,
        UnsignedLong = PY_TYPES + 41,
        LongLong = PY_TYPES + 50,
        UnsignedLongLong = PY_TYPES + 51,
        PySizeT = PY_TYPES + 60,

        Float = PY_TYPES + 70,
        Double = PY_TYPES + 71,

        Bool = PY_TYPES + 80,
        String = PY_TYPES + 90,
        PyObject = PY_TYPES + 100,

        NUMPY_TYPES = 2000,

        NUMPY_Int8 = NUMPY_TYPES + 10,
        NUMPY_UnsignedInt8 = NUMPY_TYPES + 11,
        NUMPY_Int16 = NUMPY_TYPES + 12,
        NUMPY_UnsignedInt16 = NUMPY_TYPES + 13,
        NUMPY_Int32 = NUMPY_TYPES + 14,
        NUMPY_UnsignedInt32 = NUMPY_TYPES + 15,
        NUMPY_Int64 = NUMPY_TYPES + 16,
        NUMPY_UnsignedInt64 = NUMPY_TYPES + 17,

        NUMPY_Float16 = NUMPY_TYPES + 20,
        NUMPY_Float32 = NUMPY_TYPES + 21,
        NUMPY_Float64 = NUMPY_TYPES + 22,
        NUMPY_Float128 = NUMPY_TYPES + 23,

        NUMPY_Bool = NUMPY_TYPES + 30
    };
}
// register a conversion for FFIType
namespace rynlib {
    namespace python {
        template<>
        inline PyObject* as_python<plzffi::FFIType>(plzffi::FFIType data) {
            return as_python<int>(static_cast<int>(data));
        }
        template<>
        inline plzffi::FFIType from_python<plzffi::FFIType>(PyObject* data) {
            return static_cast<plzffi::FFIType>(get_python_attr<int>(data, "value"));
        }
    }
}

namespace plzffi {

    //
    template <typename T>
    class FFITypeHandler {
        // class that helps us maintain a map between type codes & proper types
    public:
        void validate(FFIType type_code);
        T cast(FFIType type_code, std::shared_ptr<void>& data);
        PyObject* as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape);
    };
    template <typename T>
    class FFITypeHandler<T*> {
        // specialization to handle pointer types
    public:
        void validate(FFIType type_code);
        T* cast(FFIType type_code, std::shared_ptr<void>& data);
        PyObject* as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape);
    };

    //region Template Garbage
    // template shit has to be in the headers -_-
    template <typename T>
    inline void FFITypeHandler<T>::validate(FFIType type_code) {
        if (type_code != FFIType::GENERIC) {
            throw std::runtime_error("unhandled type specifier");
        }
    };
    template <typename T>
    inline void FFITypeHandler<T*>::validate(FFIType type_code) {
        FFITypeHandler<T> handler;
        handler.validate(type_code);
    }
    template <>
    void  FFITypeHandler<npy_bool>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_int8>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_int16>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_int32>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_int64>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_uint8>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_uint16>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_uint32>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_uint64>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_float16>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_float32>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_float64>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<npy_float128>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<unsigned char>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<short>::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<unsigned short >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<int >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<unsigned int >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<long >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<unsigned long >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<long long >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<unsigned long long >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<Py_ssize_t >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<float >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<double >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<bool >::validate(FFIType type_code);
    template <>
    void  FFITypeHandler<std::string >::validate(FFIType type_code);
    //endregion

    template <typename T>
    inline T FFITypeHandler<T>::cast(FFIType type_code, std::shared_ptr<void>& data) {
        validate(type_code);
        return *static_cast<T*>(data.get());
    }
    template <typename T>
    inline T* FFITypeHandler<T*>::cast(FFIType type_code, std::shared_ptr<void>& data) {
        validate(type_code);
        return static_cast<T*>(data.get());
    }

    template <typename T>
    inline PyObject* FFITypeHandler<T>::as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
        if (!shape.empty()) {
            return FFITypeHandler<T*>().as_python(type_code, data, shape);
        }
        validate(type_code);
        return rynlib::python::as_python<T>(*static_cast<T*>(data.get()));
    }
    template <typename T>
    inline PyObject* FFITypeHandler<T*>::as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
        // we use NumPy for all pointer types
        validate(type_code);
        return rynlib::python::numpy_from_data<T>(static_cast<T*>(data.get()), shape);
    }

    class FFIArgument {
        std::string param_key;
        std::vector<size_t> shape_vec; // for holding NumPy data
        FFIType type_char;
    public:
        FFIArgument(
                std::string &name,
                FFIType type,
                std::vector<size_t> &shape
        ) : param_key(name), shape_vec(shape), type_char(type) {}
        FFIArgument(
                const char* name,
                FFIType type,
                std::vector<int> shape
        ) : param_key(name), type_char(type) {
//            for (auto s : shape) { shape_vec}
            shape_vec = std::vector<size_t>(shape.begin(), shape.end());
        }
        FFIArgument() = default;

        std::string name() {return param_key;}
        std::vector<size_t> shape() {return shape_vec;}
        FFIType type() {return type_char;}

        PyObject * as_tuple() {
            return Py_BuildValue("(NNN)",
                                 rynlib::python::as_python<std::string>(param_key),
                                 rynlib::python::as_python<FFIType>(type_char),
                                 rynlib::python::as_python_tuple<size_t>(shape_vec)
                                 );
        }
    };

    class FFIParameter {
        // object that maps onto the python FFI stuff...
        PyObject *py_obj;
        FFIArgument arg_spec;
        std::shared_ptr<void> param_data; // we void pointer this to make it easier to handle
    public:
        FFIParameter(
                PyObject *obj,
                FFIArgument& arg
                ) : py_obj(obj), arg_spec(arg), param_data() {};

        FFIParameter(
                std::shared_ptr<void>& data,
                FFIArgument& arg
        ) : py_obj(NULL), arg_spec(arg), param_data(data) {};

        explicit FFIParameter(PyObject *obj) : py_obj(obj), arg_spec() { init(); }

        FFIParameter() = default;

        void init();

        std::string name() { return arg_spec.name(); }
        std::vector<size_t> shape() { return arg_spec.shape(); }
        FFIType type() { return arg_spec.type(); }

        template <typename T>
        T value() {
            FFITypeHandler<T> handler;
            return handler.cast(type(), param_data);
        }

        std::shared_ptr<void> _raw_ptr() { return param_data; } // I put this out there so people smarter than I can use it

        PyObject* as_python();
    };

    class FFIParameters {
        // object that maps onto the python FFI stuff...
        PyObject *py_obj;
        std::vector<FFIParameter> params;
    public:
        FFIParameters() : py_obj(), params() {};
        explicit FFIParameters(PyObject* param_obj) : py_obj(param_obj) {
            params = {};
            init();
        }
        void init();
        size_t param_index(std::string& key);
        FFIParameter get_parameter(std::string& key);
        FFIParameter get_parameter(const char* key);
        void set_parameter(std::string& key, FFIParameter& param);
        void set_parameter(const char* key, FFIParameter& param);

        template <typename T>
        T value(std::string& key) {
            return get_parameter(key).value<T>();
        }
        template <typename T>
        T value(const char* key) {
            return get_parameter(key).value<T>();
        }

    };

}

// register a conversion for FFIType
namespace rynlib {
    namespace python {
        template<>
        inline PyObject* as_python<plzffi::FFIParameter>(plzffi::FFIParameter data) {
            if (plzffi::debug_print()) printf("  Converting FFIParameter to PyObject...");
            return data.as_python();
        }
        template<>
        inline plzffi::FFIParameter from_python<plzffi::FFIParameter>(PyObject* data) {
            if (plzffi::debug_print()) printf("  Converting PyObject to FFIParameter...");
            return plzffi::FFIParameter(data);
        }
    }
}

#endif //RYNLIB_FFIPARAMETERS_HPP
