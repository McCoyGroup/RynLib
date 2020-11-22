
#ifndef RYNLIB_PYALLUP_HPP
#define RYNLIB_PYALLUP_HPP

#include "numpy/arrayobject.h"
#include <stdexcept>

namespace rynlib {
    namespace python {

        // do all NumPy fuckery in this file and in this file alone
        inline long _np_init() {
            if(PyArray_API == NULL)
            {
                import_array();
            }
        }

        template<typename T>
        inline T from_python(PyObject* data) {
            throw std::runtime_error("Can only handle simple python types");
        }
        template <>
        inline PyObject * from_python<PyObject *>(PyObject *data) {
            return data;
        }
        template <>
        inline Py_ssize_t from_python<Py_ssize_t>(PyObject *data) {
            return PyLong_AsSsize_t(data);
        }
        template <>
        inline size_t from_python<size_t>(PyObject *data) {
            return PyLong_AsSize_t(data);
        }
        template <>
        inline double from_python<double>(PyObject *data) {
            return PyFloat_AsDouble(data);
        }
        template <>
        inline bool from_python<bool >(PyObject *data) {
            return PyObject_IsTrue(data);
        }
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
        inline T from_python_capsule(PyObject* cap, const char* name) {
            auto obj = PyCapsule_GetPointer(cap, name);
            if (obj == NULL) {
                throw std::runtime_error("Capsule error");
            }
            return obj; // implicit cast on return
        }

        template<typename T>
        inline T* get_numpy_data(PyObject *array) {
            _np_init();
            if (!PyArray_Check(array)) {
                PyErr_SetString(PyExc_TypeError, "expected numpy array");
                throw std::runtime_error("requires NumPy array");
            }
            return (T*) PyArray_DATA(array);
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
        inline PyObject * numpy_from_data(
                T* buffer,
                std::vector<size_t> shape
        ) {
            throw std::runtime_error("unknown dtype");
        }
        template<>
        inline PyObject* numpy_from_data<Real_t >(
                RawPotentialBuffer buffer,
                std::vector<size_t> shape
        ) {
            return numpy_from_data<Real_t >(
                    buffer,
                    NPY_DOUBLE,
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
                auto val = get_python_attr_iterable<T>(attr_ob);
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
                auto val = get_python_attr_ptr<T>(attr_ob);
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
