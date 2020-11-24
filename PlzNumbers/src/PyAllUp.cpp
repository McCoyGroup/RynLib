
#include "PyAllUp.hpp"

// specify which version of the API we're on
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include "numpy/arrayobject.h"
#include <stdexcept>
#include <vector>
#include <string>

namespace rynlib {
    namespace python {

        // do all NumPy fuckery in this file and in this file alone
//        long* _np_fuckery() {
//            if(PyArray_API == NULL) {
//                import_array();
//                return NULL;
//            } else {
//                return NULL;
//            }
//        }
//        void _np_init() {
//            auto p = _np_fuckery();
//            if (p != NULL) throw std::runtime_error("NumPy failed to load");
//        }
//
//        void _check_py_arr(PyObject* array) {
//            if (!PyArray_Check(array)) {
//                PyErr_SetString(PyExc_TypeError, "expected numpy array");
//                throw std::runtime_error("requires NumPy array");
//            }
//        }
//
//        std::string get_python_repr(PyObject* obj) {
//            PyObject *repr= PyObject_Repr(obj);
//            auto rep = from_python<std::string>(repr);
//            Py_XDECREF(repr);
//            return rep;
//        }

//        template<typename T>
//        T from_python(PyObject* data) {
//            return T(data);
////            throw std::runtime_error("Can only handle simple python types");
//        }
//        template <>
//        PyObject * from_python<PyObject *>(PyObject *data) {
//            return data;
//        }
//        template <>
//        Py_ssize_t from_python<Py_ssize_t>(PyObject *data) {
//            return PyLong_AsSsize_t(data);
//        }
//        template <>
//        size_t from_python<size_t>(PyObject *data) {
//            return PyLong_AsSize_t(data);
//        }
//        template <>
//        double from_python<double>(PyObject *data) {
//            return PyFloat_AsDouble(data);
//        }
//        template <>
//        bool from_python<bool >(PyObject *data) {
//            return PyObject_IsTrue(data);
//        }
//        template <>
//        std::string from_python<std::string >(PyObject *data) {
//            // we're ditching even the pretense of python 2 support
//            PyObject* pyStr = NULL;
//            pyStr = PyUnicode_AsEncodedString(data, "utf-8", "strict");
//            if (pyStr == NULL) {
//                throw std::runtime_error("bad python shit");
//            };
//            const char *strExcType =  PyBytes_AsString(pyStr);
//            std::string str = strExcType; // data needs to be copied...will this do it?
//            Py_XDECREF(pyStr);
//            return str;
//        }
//
//        template<typename T>
//        std::vector<T> from_python_iterable(PyObject* data, Py_ssize_t num_els) {
//            std::vector<T> vec(num_els);
//            // iterate through list
//            PyObject *iterator = PyObject_GetIter(data);
//            if (iterator == NULL) {
//                throw std::runtime_error("Iteration error");
//            }
//            PyObject *item;
//            Py_ssize_t i = 0;
//            while ((item = PyIter_Next(iterator))) {
//                vec[i] = from_python<T>(item);
//                Py_DECREF(item);
//            }
//            Py_DECREF(iterator);
//            if (PyErr_Occurred()) {
//                throw std::runtime_error("Iteration error");
//            }
//
//            return vec;
//        }
//        template<typename T>
//        std::vector<T> from_python_iterable(PyObject* data) {
//            return from_python_iterable<T>(data, PyObject_Length(data));
//        }
//
//        template<typename T>
//        inline T from_python_capsule(PyObject* cap, const char* name) {
//            auto obj = PyCapsule_GetPointer(cap, name);
//            if (obj == NULL) {
//                throw std::runtime_error("Capsule error");
//            }
//            return obj; // implicit cast on return
//        }
//
//        template<typename T>
//        T* get_numpy_data(PyObject *array) {
//            _np_init();
//            _check_py_arr(array);
//            auto py_arr = (PyArrayObject*) array;
//            return (T*) PyArray_DATA(py_arr);
//        }
//        template<typename T>
//        T* from_python_buffer(PyObject* data) { // Pointer types _only_ allowed for NumPy arrays
//            return get_numpy_data<T>(data);
//        }
//
//        template<typename T>
//        PyObject * numpy_from_data(
//                T* buffer,
//                NPY_TYPES dtype,
//                std::vector<size_t> shape
//        ) {
//            _np_init();
//            PyObject *dat = PyArray_SimpleNewFromData(
//                    shape.size(),
//                    (npy_intp*) shape.data(),
//                    dtype,
//                    buffer
//            );
//            if (dat == NULL) {
//                throw std::runtime_error("bad numpy shit");
//            };
//            return dat;
//        }
//        template<typename T>
//        PyObject * numpy_from_data(
//                T* buffer,
//                std::vector<size_t> shape
//        ) {
//            throw std::runtime_error("unknown dtype");
//        }
//        template<>
//        PyObject* numpy_from_data<double>(
//                double* buffer,
//                std::vector<size_t> shape
//        ) {
//            return numpy_from_data<double>(
//                    buffer,
//                    NPY_DOUBLE,
//                    shape
//            );
//        }
//
//        template <typename T>
//        std::vector<T > numpy_shape(PyObject* obj) {
//            _np_init();
//
//            auto arr = (PyArrayObject*) obj; // I think this is how one builds an array obj...?
//            T* shp = (T*) PyArray_SHAPE(arr);
//            return std::vector<T >(shp, shp + PyArray_NDIM(arr));
//        }
//
//        template <typename T>
//        T get_python_attr(PyObject* obj, std::string& attr) {
//            auto attr_ob = get_python_attr<PyObject*>(obj, attr);
//            try {
//                auto val = from_python<T>(attr_ob);
//                Py_XDECREF(attr_ob); // annoying...
//                return val;
//            } catch (std::exception &e) {
//                Py_XDECREF(attr_ob);
//                throw e;
//            }
//        }
//        template <typename T>
//        T get_python_attr(PyObject* obj, const char* attr) {
//            std::string attr_str = attr;
//            return get_python_attr<T>(obj, attr_str);
//        }

//        template <typename T>
//        std::vector<T> get_python_attr_iterable(PyObject* obj, std::string& attr) {
//            auto attr_ob = get_python_attr<PyObject *>(obj, attr);
//            try {
//                auto val = get_python_attr_iterable<T>(attr_ob);
//                Py_XDECREF(attr_ob); // annoying...
//                return val;
//            } catch (std::exception &e) {
//                Py_XDECREF(attr_ob);
//                throw e;
//            }
//        }
//        template <typename T>
//        std::vector<T> get_python_attr_iterable(PyObject* obj, const char* attr) {
//            std::string attr_str = attr;
//            return get_python_attr_iterable<T>(obj, attr_str);
//        }

//        template <typename T>
//        T* get_python_attr_ptr(PyObject* obj, std::string& attr) {
//            auto attr_ob = get_python_attr<PyObject *>(obj, attr);
//            try {
//                auto val = get_python_attr_ptr<T>(attr_ob);
//                Py_XDECREF(attr_ob); // annoying...
//                return val;
//            } catch (std::exception &e) {
//                Py_XDECREF(attr_ob);
//                throw e;
//            }
//        }
//        template <typename T>
//        T* get_python_attr_ptr(PyObject* obj, const char* attr) {
//            std::string attr_str = attr;
//            return get_python_attr_ptr<T>(obj, attr_str);
//        }

    }

}