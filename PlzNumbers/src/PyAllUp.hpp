
#ifndef RYNLIB_PYALLUP_HPP
#define RYNLIB_PYALLUP_HPP

#include "Python.h"
#include "numpy/ndarraytypes.h"
#include <vector>
#include <string>

namespace rynlib {
    namespace python {

        template<typename T>
        T from_python(PyObject* data);
        template<typename T>
        std::vector<T> from_python_iterable(PyObject* data, Py_ssize_t num_els);
        template<typename T>
        std::vector<T> from_python_iterable(PyObject* data);
        template<typename T>
        T from_python_capsule(PyObject* cap, const char* name);
        template<typename T>
        T* get_numpy_data(PyObject *array);
        template<typename T>
        T* from_python_buffer(PyObject* data);

        template<typename T>
        PyObject * numpy_from_data(T* buffer, NPY_TYPES dtype, std::vector<size_t> shape);
        template<typename T>
        PyObject * numpy_from_data(T* buffer, std::vector<size_t> shape);

        template <typename T>
        std::vector<T > numpy_shape(PyObject* obj);

        template <typename T>
        T get_python_attr(PyObject* obj, std::string& attr);
        template <typename T>
        inline T get_python_attr(PyObject* obj, const char* attr);

        template <typename T>
        std::vector<T> get_python_attr_iterable(PyObject* obj, std::string& attr);
        template <typename T>
        std::vector<T> get_python_attr_iterable(PyObject* obj, const char* attr);

        template <typename T>
        T* get_python_attr_ptr(PyObject* obj, std::string& attr);
        template <typename T>
        T* get_python_attr_ptr(PyObject* obj, const char* attr);

    }

}

#endif //RYNLIB_PYALLUP_HPP
