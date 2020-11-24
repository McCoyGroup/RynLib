
#ifndef RYNLIB_PYALLUP_HPP
#define RYNLIB_PYALLUP_HPP

#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include <vector>
#include <string>
#include <stdexcept>

namespace rynlib {
    namespace python {

        inline long* _np_fuckery() {
            if(PyArray_API == NULL) {
                import_array();
                return NULL;
            } else {
                return NULL;
            }
        }
        inline void _np_init() {
            auto p = _np_fuckery();
            if (p != NULL) throw std::runtime_error("NumPy failed to load");
        }

        inline void _check_py_arr(PyObject* array) {
            if (!PyArray_Check(array)) {
                PyErr_SetString(PyExc_TypeError, "expected numpy array");
                throw std::runtime_error("requires NumPy array");
            }
        }

        template<typename T>
        inline T from_python(PyObject* data) {
//            auto obj = new T {data}; // memory risky...
//            return obj; // assume we have a constructor that can ingest the python type
            std::string tname = typeid(T).name();
            std::string bad_type = "For type " + tname;
            bad_type += ": can only convert python data to simple types. Define your own converter if you need it";
            throw std::runtime_error(bad_type);
        }
        template <>
        inline PyObject * from_python<PyObject *>(PyObject *data) {
            return data;
        }
        template <>
        inline char from_python<char>(PyObject *data) { return PyLong_AsSsize_t(data); }
        template <>
        inline unsigned char from_python<unsigned char>(PyObject *data) { return PyLong_AsSsize_t(data); }
        template <>
        inline int from_python<int>(PyObject *data) { return PyLong_AsSsize_t(data); }
        template <>
        inline unsigned int from_python<unsigned int>(PyObject *data) { return PyLong_AsSsize_t(data); }
        template <>
        inline short from_python<short>(PyObject *data) { return PyLong_AsSsize_t(data); }
        template <>
        inline unsigned short from_python<unsigned short>(PyObject *data) { return PyLong_AsSsize_t(data); }
        template <>
        inline long from_python<long>(PyObject *data) { return PyLong_AsSsize_t(data); }
        template <>
        inline unsigned long from_python<unsigned long>(PyObject *data) { return PyLong_AsSsize_t(data); }
        template <>
        inline long long from_python<long long>(PyObject *data) { return PyLong_AsSsize_t(data); }
        template <>
        inline unsigned long long from_python<unsigned long long>(PyObject *data) { return PyLong_AsSsize_t(data); }
//        template <>
//        inline Py_ssize_t from_python<Py_ssize_t>(PyObject *data) { return PyLong_AsSsize_t(data);  }
//        template <>
//        inline size_t from_python<size_t>(PyObject *data) { return PyLong_AsSize_t(data); }
        template <>
        inline float from_python<float>(PyObject *data) { return PyFloat_AsDouble(data); }
        template <>
        inline double from_python<double>(PyObject *data) { return PyFloat_AsDouble(data); }
        template <>
        inline bool from_python<bool >(PyObject *data) { return PyObject_IsTrue(data); }
        template <>
        inline std::string from_python<std::string >(PyObject *data) {
            // we're ditching even the pretense of python 2 support
            PyObject* pyStr = NULL;
            pyStr = PyUnicode_AsEncodedString(data, "utf-8", "strict");
            if (pyStr == NULL) {
                throw std::runtime_error("bad python shit");
            };
            const char *strExcType =  PyBytes_AsString(pyStr);
            std::string str = strExcType; // data needs to be copied...will this do it?
            Py_XDECREF(pyStr);
            return str;
        }

        template<typename T>
        inline PyObject* as_python(T data) {
            std::string tname = typeid(T).name(); // often garbage, but we use this just for debug purposes
            std::string bad_type = "For type " + tname;
            bad_type += ": can only convert simple types to python data. Define your own converter if you need it";
            throw std::runtime_error(bad_type);
        }
        template <>
        inline PyObject * as_python<PyObject *>(PyObject *data) {
            return data;
        }
        template <>
        inline PyObject *as_python<char>(char data) { return Py_BuildValue("b", data); }
        template <>
        inline PyObject *as_python<unsigned char>(unsigned char data) { return Py_BuildValue("B", data); }
        template <>
        inline PyObject *as_python<short>(short data) { return Py_BuildValue("h", data); }
        template <>
        inline PyObject *as_python<unsigned short>(unsigned short data) { return Py_BuildValue("H", data); }
        template <>
        inline PyObject *as_python<int>(int data) { return Py_BuildValue("i", data); }
        template <>
        inline PyObject *as_python<unsigned int>(unsigned int data) { return Py_BuildValue("I", data); }
        template <>
        inline PyObject *as_python<long>(long data) { return Py_BuildValue("l", data); }
        template <>
        inline PyObject *as_python<unsigned long>(unsigned long data) { return Py_BuildValue("k", data); }
        template <>
        inline PyObject *as_python<long long>(long long data) { return Py_BuildValue("L", data); }
        template <>
        inline PyObject *as_python<unsigned long long>(unsigned long long data) { return Py_BuildValue("K", data); }
//        template <>
//        inline Py_ssize_t from_python<Py_ssize_t>(PyObject *data) { return PyLong_AsSsize_t(data);  }
//        template <>
//        inline size_t from_python<size_t>(PyObject *data) { return PyLong_AsSize_t(data); }
        template <>
        inline PyObject *as_python<float>(float data) { return Py_BuildValue("f", data); }
        template <>
        inline PyObject *as_python<double>(double data) { return Py_BuildValue("d", data); }
        template <>
        inline PyObject * as_python<std::string>(std::string data) {
            return Py_BuildValue("s", data.c_str());
        }
        template <>
        inline PyObject * as_python<const char*>(const char* data) {
            return Py_BuildValue("s", data);
        }

        template<typename T>
        inline std::vector<T> from_python_iterable(PyObject* data, Py_ssize_t num_els) {
            std::vector<T> vec(num_els);
            // iterate through list
            PyObject *iterator = PyObject_GetIter(data);
            if (iterator == NULL) {
                throw std::runtime_error("Iteration error");
            }
            PyObject *item;
            Py_ssize_t i = 0;
            while ((item = PyIter_Next(iterator))) {
                vec[i] = from_python<T>(item);
                Py_DECREF(item);
            }
            Py_DECREF(iterator);
            if (PyErr_Occurred()) {
                throw std::runtime_error("Iteration error");
            }

            return vec;
        }
        template<typename T>
        inline std::vector<T> from_python_iterable(PyObject* data) {
            return from_python_iterable<T>(data, PyObject_Length(data));
        }

        template<typename T>
        inline PyObject *as_python_tuple(std::vector<T> data, Py_ssize_t num_els) {
            auto tup = PyTuple_New(num_els);
            for (size_t i = 0; i < data.size(); i++) {
                PyTuple_SET_ITEM(tup, i, as_python<T>(data[i]));
            }
            return tup;
        }
        template<typename T>
        inline PyObject *as_python_tuple(std::vector<T> data) {
            return as_python_tuple<T>(data, data.size());
        }

        inline std::string get_python_repr(PyObject* obj) {
            PyObject *repr= PyObject_Repr(obj);
            auto rep = from_python<std::string>(repr);
            Py_XDECREF(repr);
            return rep;
        }

        template<typename T>
        inline T get_pycapsule_ptr(PyObject* cap, const char* name) {
            auto obj = PyCapsule_GetPointer(cap, name);
            if (obj == NULL) {
                throw std::runtime_error("Capsule error");
            }
            return T(obj); // explicit cast
        }
        template<typename T>
        inline T get_pycapsule_ptr(PyObject* cap, std::string& name) {
            return get_pycapsule_ptr<T>(cap, name.c_str());
        }
        template<typename T>
        inline T get_pycapsule_ptr(PyObject* cap, std::string name) {
            return get_pycapsule_ptr<T>(cap, name.c_str());
        }
        template<typename T>
        inline T from_python_capsule(PyObject* cap, const char* name) {
            return *get_pycapsule_ptr<T*>(cap, name); // explicit dereference
        }
        template<typename T>
        inline T from_python_capsule(PyObject* cap, std::string& name) {
            return from_python_capsule<T>(cap, name.c_str());
        }
        template<typename T>
        inline T from_python_capsule(PyObject* cap, std::string name) {
            return from_python_capsule<T>(cap, name.c_str());
        }

        template<typename T>
        inline T* get_numpy_data(PyObject *array) {
            _np_init();
            _check_py_arr(array);
            auto py_arr = (PyArrayObject*) array;
            return (T*) PyArray_DATA(py_arr);
        }
        template<typename T>
        inline T* from_python_buffer(PyObject* data) { // Pointer types _only_ allowed for NumPy arrays
            return get_numpy_data<T>(data);
        }

        template<typename T>
        inline PyObject * numpy_from_data(
                T* buffer,
                NPY_TYPES dtype,
                std::vector<size_t> shape
        ) {
            _np_init();

            auto nd = shape.size();
            auto dims = (npy_intp*) shape.data();
//            printf("huh fack %lu %lu %lu %lu\n", dims[0], dims[1], dims[2], dims[3]);
            auto data = (void*)buffer;
            PyObject *dat = PyArray_SimpleNewFromData(
                    nd,
                    dims,
                    dtype,
                    data
            );

//            printf("huh fack\n");
            if (dat == NULL) {
                throw std::runtime_error("bad numpy shit");
            }
//            else {
//                printf("huh fack2...\n");
//                throw std::runtime_error("good numpy shit");
//            }

            return dat;
        }
        template<typename T>
        inline PyObject* numpy_from_data(
                T* buffer,
                std::vector<size_t> shape
        ) {
            throw std::runtime_error("unknown dtype");
        }
        template<>
        inline PyObject* numpy_from_data<double>(
                double* buffer,
                std::vector<size_t> shape
        ) {

//            auto goddamnit = (npy_intp*) shape.data();
//            size_t num_args = 1;
//            for (unsigned long p : shape) {
//                printf("%lu", p);
//                num_args *= p;
//            }
//            printf("....okay?\n");
//            printf("huuuuh %lu %p %s\n", num_args, buffer, (num_args > 0) ? "wtttttf" : "noooooh");
//            printf("huuuuh %f, %f, %f, %f\n", buffer[0], buffer[1], buffer[2], buffer[3]);
//            size_t q = 0;
//            printf("huuuuh %s \n", (num_args > q) ? "yes?" : "oh okay no");
//            q = 20;
//            printf("huuuuh %s \n", (num_args < q) ? "yes?" : "oh okay no");
//            q = 10;
//            printf("huuuuh %s \n", (num_args < q) ? "yes?" : "oh okay no");
//            q = 18;
//            printf("huuuuh %s \n", (num_args == q) ? "yes?" : "oh okay no");
//            for (q = 0; q < num_args; q++) {
//                printf("wat ");
//            }
//            for (q = 0; q < num_args; q++) {
//                printf("%f...", buffer[q]);
//            }
//            printf("huuuuh %s \n", (num_args == q) ? "yes?" : "oh okay no");
//
//            throw std::runtime_error("...?");

            return numpy_from_data<double>(
                    buffer,
                    NPY_FLOAT64,
                    shape
            );
        }

        template <typename T>
        inline std::vector<T > numpy_shape(PyObject* obj) {
            _np_init();

            auto arr = (PyArrayObject*) obj; // I think this is how one builds an array obj...?
            T* shp = (T*) PyArray_SHAPE(arr);
            return std::vector<T >(shp, shp + PyArray_NDIM(arr));
        }

        template <typename T>
        inline T get_python_attr(PyObject* obj, std::string& attr) {
            auto attr_ob = get_python_attr<PyObject*>(obj, attr);
            try {
                auto val = from_python<T>(attr_ob);
                Py_XDECREF(attr_ob); // annoying...
                return val;
            } catch (std::exception &e) {
                Py_XDECREF(attr_ob);
                throw e;
            }
        }
        template<>
        inline PyObject* get_python_attr<PyObject *>(PyObject* obj, std::string& attr) {
            if (obj == NULL) {
                std::string err = "requested attrib " + attr + " from NULL object";
                PyErr_SetString(
                        PyExc_TypeError,
                        err.c_str()
                        );
                throw std::runtime_error("Python issues");
            }
            PyObject* ret = PyObject_GetAttrString(obj, attr.c_str());
            if (ret == NULL) {
                throw std::runtime_error("Python issues");
            }
            return ret;
        }
        template <typename T>
        inline T get_python_attr(PyObject* obj, const char* attr) {
            std::string attr_str = attr;
            return get_python_attr<T>(obj, attr_str);
        }


        template <typename T>
        inline std::vector<T> get_python_attr_iterable(PyObject* obj, std::string& attr) {
            auto attr_ob = get_python_attr<PyObject *>(obj, attr);
            try {
                auto val = from_python_iterable<T>(attr_ob);
                Py_XDECREF(attr_ob); // annoying...
                return val;
            } catch (std::exception &e) {
                Py_XDECREF(attr_ob);
                throw e;
            }
        }
        template <typename T>
        inline std::vector<T> get_python_attr_iterable(PyObject* obj, const char* attr) {
            std::string attr_str = attr;
            return get_python_attr_iterable<T>(obj, attr_str);
        }


        template <typename T>
        inline T* get_python_attr_ptr(PyObject* obj, std::string& attr) {
            auto attr_ob = get_python_attr<PyObject *>(obj, attr);
            try {
                auto val = from_python_buffer<T>(attr_ob);
                Py_XDECREF(attr_ob); // annoying...
                return val;
            } catch (std::exception &e) {
                Py_XDECREF(attr_ob);
                throw e;
            }
        }
        template <typename T>
        inline T* get_python_attr_ptr(PyObject* obj, const char* attr) {
            std::string attr_str = attr;
            return get_python_attr_ptr<T>(obj, attr_str);
        }

    }

}

#endif //RYNLIB_PYALLUP_HPP
