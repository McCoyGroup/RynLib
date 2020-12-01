
#ifndef RYNLIB_FFIPARAMETERS_HPP
#define RYNLIB_FFIPARAMETERS_HPP

#include "PyAllUp.hpp"
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <type_traits>
#include <algorithm>

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
namespace rynlib::python {
        template<>
        inline PyObject* as_python<plzffi::FFIType>(plzffi::FFIType data) {
            return as_python<int>(static_cast<int>(data));
        }
        template<>
        inline plzffi::FFIType from_python<plzffi::FFIType>(PyObject* data) {
            return static_cast<plzffi::FFIType>(get_python_attr<int>(data, "value"));
        }
}

namespace plzffi {

    // we set up an object that can store a sequence
    // of FFITypes
    template <FFIType...ffi_types>
    struct FFITypeTuple {
    };

    template <typename T, FFIType...ffi_types>
    struct FFIValidTypeset{
        using type=T;
        constexpr static const auto valid = std::make_tuple(ffi_types...);
    };

    using FFITypesets = std::tuple<
            FFIValidTypeset<npy_bool,
                    FFIType::NUMPY_Bool, FFIType::NUMPY_UnsignedInt8, FFIType::UnsignedChar>,
            FFIValidTypeset<npy_int8, FFIType::NUMPY_Int8>,
            FFIValidTypeset<npy_int16, FFIType::Short, FFIType::NUMPY_Int16>,
            FFIValidTypeset<npy_int32, FFIType::NUMPY_Int32, FFIType::Int>,
            FFIValidTypeset<npy_int64, FFIType::NUMPY_Int64, FFIType::Long, FFIType::PySizeT>,
            FFIValidTypeset<npy_uint16,
                    FFIType::NUMPY_UnsignedInt16, FFIType::NUMPY_Float16, FFIType::UnsignedShort>,
            FFIValidTypeset<npy_uint32, FFIType::NUMPY_UnsignedInt32, FFIType::UnsignedInt>,
            FFIValidTypeset<npy_uint64, FFIType::NUMPY_UnsignedInt64, FFIType::UnsignedLong>,
            FFIValidTypeset<npy_float32, FFIType::NUMPY_Float32, FFIType::Float>,
            FFIValidTypeset<npy_float64, FFIType::NUMPY_Float64, FFIType::Double>,
            FFIValidTypeset<npy_float128, FFIType::NUMPY_Float128>,
            FFIValidTypeset<long long, FFIType::LongLong>,
            FFIValidTypeset<unsigned long long, FFIType::UnsignedLongLong>,
            FFIValidTypeset<bool, FFIType::Bool>,
            FFIValidTypeset<std::string, FFIType::String>
    >;

    template <FFIType F, typename T>
    struct FFITypePair {
        // thank you again StackOverflow
        static const decltype(F) value = F;
        using type=T;
    };
    // define the mapping between FFIType and true types
    using FFITypePairs = std::tuple<
            FFITypePair<FFIType::PyObject, PyObject*>,
            FFITypePair<FFIType::UnsignedChar, unsigned char>,
            FFITypePair<FFIType::Short, short>,
            FFITypePair<FFIType::UnsignedShort, unsigned short>,
            FFITypePair<FFIType::Int, int>,
            FFITypePair<FFIType::UnsignedInt, unsigned int>,
            FFITypePair<FFIType::Long, long>,
            FFITypePair<FFIType::UnsignedLong, unsigned long>,
            FFITypePair<FFIType::LongLong, long long>,
            FFITypePair<FFIType::UnsignedLongLong, unsigned long long>,
            FFITypePair<FFIType::PySizeT, Py_ssize_t>,
            FFITypePair<FFIType::Float, float>,
            FFITypePair<FFIType::Double, double>,
            FFITypePair<FFIType::Bool, bool>,
            FFITypePair<FFIType::String, std::string>,
            FFITypePair<FFIType::NUMPY_Bool, npy_bool>,
            FFITypePair<FFIType::NUMPY_Int8, npy_int8>,
            FFITypePair<FFIType::NUMPY_Int16, npy_int16>,
            FFITypePair<FFIType::NUMPY_Int32, npy_int32>,
            FFITypePair<FFIType::NUMPY_Int64, npy_int64>,
            FFITypePair<FFIType::NUMPY_UnsignedInt8, npy_uint8>,
            FFITypePair<FFIType::NUMPY_UnsignedInt16, npy_uint16>,
            FFITypePair<FFIType::NUMPY_UnsignedInt32, npy_uint32>,
            FFITypePair<FFIType::NUMPY_UnsignedInt64, npy_uint64>,
            FFITypePair<FFIType::NUMPY_Float16, npy_float16>,
            FFITypePair<FFIType::NUMPY_Float32, npy_float32>,
            FFITypePair<FFIType::NUMPY_Float64, npy_float64>,
            FFITypePair<FFIType::NUMPY_Float128, npy_float128>
    >;



    template <typename, typename...>
    struct FFITypeValidator;
    template <typename T>
    struct FFITypeValidator<T> {
        static void validate (FFIType type) {
            std::string msg = "ERROR: unhandled typename/FFIType pair";
            std::string tname = typeid(T).name();
            msg += "(" + tname + "/" + std::to_string(static_cast<int>(type)) + ")";
            printf("%s\n", msg.c_str());
            throw std::runtime_error(msg);
        }
        static FFIType typecode() {
            std::string msg = "ERROR: unhandled typename";
            msg += typeid(T).name();
            printf("%s\n", msg.c_str());
            throw std::runtime_error(msg);
        }
    };
    template <typename D, typename T, typename... Args> // expects FFITypeset objects
    struct FFITypeValidator<D, T, Args...> {
        static void validate (FFIType type) {
            if (std::is_same<D, typename T::type>::value) {
//                auto valids = T::valid;
//                auto nels = std::tuple_size_v<decltype(T::valid)>;
                // https://stackoverflow.com/a/40233852/5720002
                // Functional-programming loop over tuple elements
                // to check if the type is in the held tuple
                // This is basically a bunch of template fuckery
                // just to avoid a huge switch...I think it's worth it
                bool found = std::apply(
                        [type](auto &&... args) {
                            return ( (decltype(args)(args) == type) || ...);
                        },
                        T::valid
                );
                if ( !found ) {
                    throw std::runtime_error("typename/FFIType mismatch");
                }
            } else {
                FFITypeValidator<D, Args...>::validate(type);
            }
        }
        static FFIType typecode() {
            if (std::is_same<D, typename T::type>::value) {
                return T::value;
            } else {
                return FFITypeValidator<D, Args...>::typecode();
            }
        }
    };

    template <typename T, size_t... Idx>
    inline void validate_type(FFIType type, std::index_sequence<Idx...>) {
        return FFITypeValidator<T, std::tuple_element_t<Idx, FFITypesets>...>::validate(type);
    }
    template <typename T>
    inline void validate_type(FFIType type) {
        return validate_type<T>(type, std::make_index_sequence<std::tuple_size<FFITypesets>{}>{});
    }

    template <typename T, size_t... Idx>
    inline FFIType ffi_typecode(std::index_sequence<Idx...>) {
        return FFITypeValidator<T, std::tuple_element_t<Idx, FFITypePairs>...>::typecode();
    }
    template <typename T>
    inline FFIType ffi_typecode() {
        return ffi_typecode<T>(std::make_index_sequence<std::tuple_size<FFITypePairs>{}>{});
    }

    //
    template <typename T>
    class FFITypeHandler {
        // class that helps us maintain a map between type codes & proper types
    public:
        FFIType ffi_type() { return ffi_typecode<T>(); }
        void validate(FFIType type_code) { validate_type<T>(type_code); }
        T cast(FFIType type_code, std::shared_ptr<void> &data) {
            validate(type_code);
            return *static_cast<T *>(data.get());
        }
        PyObject *as_python(FFIType type_code, std::shared_ptr<void> &data, std::vector<size_t> &shape);
    };

    template <typename T>
    class FFITypeHandler<std::vector<T> > {
        // specialization to handle vector types
    public:
        FFIType ffi_type() {return ffi_typecode<T>();}
        void validate(FFIType type_code) { validate_type<T>(type_code); }
        std::vector<T> cast(FFIType type_code, std::shared_ptr<void>& data) {
            validate(type_code);
            return *static_cast<std::vector<T>>(data.get());
        }
        PyObject* as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape);
    };
    template <typename T>
    class FFITypeHandler<T*> {
        // specialization to handle pointer types
    public:
        FFIType ffi_type() {return ffi_typecode<T>();}
        void validate(FFIType type_code) { validate_type<T>(type_code); }
        T* cast(FFIType type_code, std::shared_ptr<void>& data) {
            validate(type_code);
            return static_cast<T*>(data.get());
        }
        PyObject* as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape);
    };

    template <typename T>
    inline PyObject* FFITypeHandler<T>::as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
        if (!shape.empty()) {
            return FFITypeHandler<T*>().as_python(type_code, data, shape);
        }
        validate(type_code);
        return rynlib::python::as_python<T>(*static_cast<T*>(data.get()));
    }
    template <typename T>
    inline PyObject* FFITypeHandler<std::vector<T> >::as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
        validate(type_code);
        return rynlib::python::as_python<std::vector<T> >(*static_cast<std::vector<T>*>(data.get()));
    }
    template <typename T>
    inline PyObject* FFITypeHandler<T*>::as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
        // we use NumPy for all pointer types
        validate(type_code);
        return rynlib::python::numpy_from_data<T>(static_cast<T*>(data.get()), shape);
    }

    // recursive definition to loop through and test all the type pairs (thank you StackOverflow)
    // we'll embed most conversion functions here so that we don't need to duplicate the boiler plate
    template <typename...>
    class FFIConversionManager;
    template<>
    class FFIConversionManager<> {
    public:
        static std::shared_ptr<void> from_python_attr(FFIType type, PyObject* py_obj, const char* attr, std::vector<size_t>& shape) {
            std::string garb = "unhandled type specifier in converting from python: " + std::to_string(static_cast<int>(type));
            throw std::runtime_error(garb.c_str());
        }
        static PyObject* as_python(FFIType type,  std::shared_ptr<void>& data, std::vector<size_t>& shape) {
            std::string garb = "unhandled type specifier in converting to python: " + std::to_string(static_cast<int>(type));
            throw std::runtime_error(garb.c_str());
        }
    };
    template <typename T, typename... Args> // expects FFITypePair objects
    class FFIConversionManager<T, Args...> {
        static const int num_ptrtypes = 13;
        constexpr static const FFIType pointer_types[13] = {
                FFIType::PyObject,
                FFIType::NUMPY_Bool,
                FFIType::NUMPY_Int8,
                FFIType::NUMPY_Int16,
                FFIType::NUMPY_Int32,
                FFIType::NUMPY_Int64,
                FFIType::NUMPY_UnsignedInt8,
                FFIType::NUMPY_UnsignedInt16,
                FFIType::NUMPY_UnsignedInt32,
                FFIType::NUMPY_UnsignedInt64,
                FFIType::NUMPY_Float16,
                FFIType::NUMPY_Float32,
                FFIType::NUMPY_Float64
        };
    public:
        static std::shared_ptr<void> from_python_attr(FFIType type, PyObject* py_obj, const char* attr, std::vector<size_t>& shape) {
            if (type == T::value) {
                auto pos = std::find(pointer_types, pointer_types + num_ptrtypes, type);
                if (pos < pointer_types + num_ptrtypes || !shape.empty()) {
                    return std::shared_ptr<void>(
                            rynlib::python::get_python_attr_ptr<typename T::type>(py_obj, attr),
                            [](typename T::type* val){delete val;}
                    );
                } else {
                    return std::make_shared<typename T::type>(
                            rynlib::python::get_python_attr<typename T::type>(py_obj, attr)
                            );
                }
            } else {
                return FFIConversionManager<Args...>::from_python_attr(type, py_obj, attr, shape);
            }
        }
        static PyObject* as_python(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
            if (type == T::value) {
                auto pos = std::find(pointer_types, pointer_types + num_ptrtypes, type);
                if (pos < pointer_types + num_ptrtypes ) {
                    return FFITypeHandler<typename T::type>().as_python(type, data, shape);
                } else {
                    return FFITypeHandler<typename T::type*>().as_python(type, data, shape);
                }
            } else {
                return FFIConversionManager<Args...>::as_python(type, data, shape);
            }
        }
    };
    template <size_t... Idx>
    inline PyObject* ffi_to_python(FFIType type, std::shared_ptr<void>&data, std::vector<size_t>& shape,
                                   std::index_sequence<Idx...> inds) {
        return FFIConversionManager<std::tuple_element_t<Idx, FFITypePairs>...>::as_python(type, data, shape);
    }
    inline PyObject* ffi_to_python(FFIType type,  std::shared_ptr<void>& data, std::vector<size_t>& shape) {
        return ffi_to_python(type, data, shape,
                             std::make_index_sequence<std::tuple_size<FFITypePairs>{}>{});
    }
    template <size_t... Idx>
    inline std::shared_ptr<void> ffi_from_python_attr(FFIType type, PyObject* obj, const char* attr,
                                                      std::vector<size_t>& shape,
                                                      std::index_sequence<Idx...> inds) {
        return FFIConversionManager<std::tuple_element_t<Idx, FFITypePairs>...>::from_python_attr(type, obj, attr, shape);
    }
    inline std::shared_ptr<void> ffi_from_python_attr(FFIType type, PyObject* obj, const char* attr,
                                                      std::vector<size_t>& shape) {
        return ffi_from_python_attr(type, obj, attr, shape,
                                    std::make_index_sequence<std::tuple_size<FFITypePairs>{}>{});
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
        // default trivial constructory
        FFIArgument() = default;
//        // default copy constructor
//        FFIArgument(const FFIArgument& arg)  : param_key(arg.param_key), shape_vec(arg.shape_vec), type_char(arg.type_char) {
//            printf("Copying FFIArgument %s\n", repr().c_str());
//        };
//        // trivial destructor should be safe?
//        ~FFIArgument() = default;


        std::string name() {return param_key;}
        std::vector<size_t> shape() {return shape_vec;}
        FFIType type() {return type_char;}

        PyObject * as_tuple();
        std::string repr();

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

        explicit FFIParameter(PyObject *obj) : py_obj(obj) { init(); }
        // default trivial constructor
        FFIParameter() = default;
//        // default copy constructor
//        FFIParameter(const FFIParameter& arg) : py_obj(arg.py_obj), arg_spec(arg.arg_spec), param_data(arg.param_data) {
//                printf("Copying FFIParameter with value %s\n", repr().c_str());
//        };
//        // trivial destructor should be safe?
//        ~FFIParameter() = default;
//        // ...?

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
        std::string repr();
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
        T value(std::string& key) { return get_parameter(key).value<T>(); }
        template <typename T>
        T value(const char* key) { return get_parameter(key).value<T>(); }
        std::vector<size_t> shape(std::string& key);
        std::vector<size_t> shape(const char* key);
        FFIType typecode(std::string& key);
        FFIType typecode(const char* key);

    };

}

// register a conversion for FFIType
namespace rynlib::python {
    template<>
    inline PyObject *as_python<plzffi::FFIParameter>(plzffi::FFIParameter data) {
        if (plzffi::debug_print()) printf("Converting FFIParameter to PyObject...\n");
        return data.as_python();
    }

    template<>
    inline plzffi::FFIParameter from_python<plzffi::FFIParameter>(PyObject *data) {
        if (plzffi::debug_print()) printf("Converting PyObject to FFIParameter...\n");
        return plzffi::FFIParameter(data);
    }
}

#endif //RYNLIB_FFIPARAMETERS_HPP
