
#ifndef SIMPLEFFI_PYALLUP_HPP

#include "Python.h"
#include "numpy/arrayobject.h"
#include <exception>
#include <string>

namespace rynlib {
    namespace python {

        // do all NumPy fuckery in this file and in this file alone
        long _np_init() {
            if(PyArray_API == NULL)
            {
                import_array();
            }
        };

        template<typename T>
        T from_python(PyObject* data) {
            throw std::runtime_error("Can only handle simple python types");
        }
        template <>
        PyObject * from_python<PyObject *>(PyObject *data) {
            return data;
        }
        template <>
        Py_ssize_t from_python<Py_ssize_t>(PyObject *data) {
            return PyLong_AsSsize_t(data);
        }
        template <>
        double from_python<double >(PyObject *data) {
            return PyFloat_AsDouble(data);
        }
        template <>
        bool from_python<bool >(PyObject *data) {
            return PyObject_IsTrue(data);
        }
        template <>
        std::string from_python<std::string >(PyObject *data) {
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
        std::vector<T> from_python_iterable(PyObject* data, Py_ssize_t num_els) {
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
        std::vector<T> from_python_iterable(PyObject* data) {
            return from_python_iterable<T>(data, PyObject_Length(data));
        }

        template<typename T>
        T from_python_capsule(PyObject* cap, const char* name) {
            auto obj = PyCapsule_GetPointer(cap, name);
            if (obj == NULL) {
                throw std::runtime_error("Capsule error");
            }
            return obj; // implicit cast on return
        }

        template<typename T>
        T* get_numpy_data(PyObject *array) {
            return PyArray_DATA(array);
        }

        template<typename T>
        PyObject * numpy_from_data(
                T* buffer,
                NPY_TYPES dtype,
                std::vector<size_t> shape
        ) {
            PyObject *dat = PyArray_SimpleNewFromData(
                    shape.size(),
                    (npy_intp*) shape.data(),
                    dtype,
                    buffer
                    );
            if (dat == NULL) {
                throw std::runtime_error("bad numpy shit");
            };
            return dat;
        }
        template<typename T>
        PyObject * numpy_from_data(
                T* buffer,
                std::vector<size_t> shape
        ) {
            throw std::runtime_error("unknown dtype");
        }
        template<>
        PyObject* numpy_from_data<double >(
                double* buffer,
                std::vector<size_t> shape
        ) {
            return numpy_from_data<double >(
                    buffer,
                    NPY_DOUBLE,
                    shape
                    );
        }

        std::vector<Py_ssize_t > numpy_shape(PyObject* obj) {
            _np_init();
            auto arr = (PyArrayObject*) obj;
            npy_intp* shp = PyArray_SHAPE(arr);
            return std::vector<Py_ssize_t >(shp, shp + PyArray_NDIM(arr));
        }
    }

}
#define SIMPLEFFI_PYALLUP_HPP

#endif //RYNLIB_PYALLUP_HPP
