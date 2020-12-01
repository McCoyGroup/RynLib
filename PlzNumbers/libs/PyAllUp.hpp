
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

        class PyallupDebugFuckery {
            bool debug_active;
        public:
            bool debug_print() {
                return debug_active;
            }
            void set_debug_print(bool db) {
                debug_active = db;
            }
        };

        inline void py_printf(const char* fmt, ...) {
            va_list args;
            va_start(args, fmt);
//            printf("PRINTING: ");
            auto print_err = vprintf(fmt, args);
            va_end(args);
            if (print_err <= 0) {
                std::string msg = "printf shat the bed with ferror = " + std::to_string(print_err);
                PyErr_SetString(
                        PyExc_SystemError,
                        msg.c_str()
                        );
                throw std::runtime_error(msg);
            }
//            printf("...?");
//            printf(" (got %d) \n", print_err);
        }

        static PyallupDebugFuckery pyadeeb;

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

        template <typename T>
        class from_python_error : public std::runtime_error {
            const char* typestr = typeid(T).name();
        public:
            from_python_error() : std::runtime_error(typestr) {};
            std::string type_id() { return typestr; }
            const char * what() const noexcept override { return typestr; }
        };

        template<typename T>
        inline T from_python(PyObject* data) {
            from_python_error<T> err;
            py_printf("WARNING: Failed to convert from python for dtype %s\n", err.type_id().c_str());
            throw from_python_error<T>();
        }
        template<typename T>
        inline T from_python(PyObject* data, const char* typechar) {
            PyObject* argtup = PyTuple_Pack(1, data);
            T val;
            int successy = PyArg_ParseTuple(argtup, typechar, &val);
            Py_XDECREF(argtup);
            if (!successy || PyErr_Occurred()) throw std::runtime_error("failed to build value...");
            return val;
        }
        template <>
        inline PyObject * from_python<PyObject *>(PyObject *data) {
            return data;
        }
        template <>
        inline int from_python<int>(PyObject *data) { return from_python<int>(data, "i"); }
        template <>
        inline char from_python<char>(PyObject *data) { return from_python<int>(data); } // meh
        template <>
        inline unsigned char from_python<unsigned char>(PyObject *data) { return from_python<unsigned char>(data, "b"); }
        template <>
        inline unsigned int from_python<unsigned int>(PyObject *data) { return from_python<unsigned int>(data, "I"); }
        template <>
        inline short from_python<short>(PyObject *data) { return from_python<short>(data, "h"); }
        template <>
        inline unsigned short from_python<unsigned short>(PyObject *data) { return from_python<unsigned short>(data, "H"); }
        template <>
        inline long from_python<long>(PyObject *data) { return from_python<long>(data, "l"); }
        template <>
        inline unsigned long from_python<unsigned long>(PyObject *data) { return from_python<unsigned long>(data, "k"); }
        template <>
        inline long long from_python<long long>(PyObject *data) { return from_python<long long>(data, "L"); }
        template <>
        inline unsigned long long from_python<unsigned long long>(PyObject *data) { return from_python<unsigned long long>(data, "K"); }
        template <>
        inline float from_python<float>(PyObject *data) { return from_python<float>(data, "f"); }
        template <>
        inline double from_python<double>(PyObject *data) { return from_python<double>(data, "d"); }
        template <>
        inline bool from_python<bool>(PyObject *data) { return from_python<int>(data, "p"); }
        template <>
        inline std::string from_python<std::string >(PyObject *data) {
            // we're ditching even the pretense of python 2 support
            PyObject* pyStr = NULL;
            pyStr = PyUnicode_AsEncodedString(data, "utf-8", "strict");
            if (pyStr == NULL) { throw std::runtime_error("bad python shit"); };
            const char *strExcType =  PyBytes_AsString(pyStr);
            std::string str = strExcType; // data needs to be copied...will this do it?
            Py_XDECREF(pyStr);
            return str;
        }

        inline std::string get_python_repr(PyObject* obj) {
            PyObject *repr= PyObject_Repr(obj);
            auto rep = from_python<std::string>(repr);
            Py_XDECREF(repr);
            return rep;
        }

        inline void print_obj(const char* fmt, PyObject* obj) {
            auto garb = get_python_repr(obj);
            py_printf(fmt, garb.c_str());
        }

        template <typename T>
        class as_python_error : public std::runtime_error {
                const char* typestr = typeid(T).name();
        public:
            as_python_error() : std::runtime_error(typestr) {};
            std::string type_id() { return typestr; }
            const char * what() const noexcept override { return typestr; }
        };

        template<typename T>
        inline PyObject* as_python(T data) {
            as_python_error<T> err;
            if (pyadeeb.debug_print()) py_printf("ERROR: failed to convert to python for dtype %s\n", err.type_id().c_str());
            throw as_python_error<T>();
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
        inline PyObject *as_python<bool>(bool data) {
            if (data) {
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
        }
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
                if (pyadeeb.debug_print()) py_printf("    > getting item %lu\n", i);
                if (PyErr_Occurred()) {
                    if (pyadeeb.debug_print()) py_printf("      ...and we failed somehow\n");
                    throw std::runtime_error("Iteration error");
                }
                if (pyadeeb.debug_print()) py_printf("    > converting from python types\n");
                vec[i] = from_python<T>(item);
                Py_DECREF(item);
                i+=1;
            }
            Py_DECREF(iterator);
            if (PyErr_Occurred()) {
                throw std::runtime_error("Iteration error");
            }
            if (i < num_els) {
                std::string msg =
                        "object was expected to have length "
                        + std::to_string(num_els) +
                        " but got " + std::to_string(i) + "elements";
                PyErr_SetString(
                        PyExc_ValueError,
                        msg.c_str()
                        );
                throw std::runtime_error("Iteration error");
            }

            return vec;
        }
        template<typename T>
        inline std::vector<T> from_python_iterable(PyObject* data) {
            auto num_els = PyObject_Size(data);
            if (num_els == -1) {
                throw std::runtime_error("size issues in iterable");
            }
            return from_python_iterable<T>(data, num_els);
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

        template <typename T>
        class numpy_type_error : public std::runtime_error {
            const char* typestr = typeid(T).name();
        public:
            numpy_type_error() : std::runtime_error(typestr) {};
            std::string type_id() { return typestr; }
        };

        template<typename T>
        inline NPY_TYPES numpy_type() {
               numpy_type_error<T> err;
               if (pyadeeb.debug_print()) py_printf("ERROR: failed to convert to python for dtype %s\n", err.type_id().c_str());
               throw numpy_type_error<T>();
            };

        template <>
        inline NPY_TYPES numpy_type<npy_bool>() { return NPY_BOOL; }
        template <>
        inline NPY_TYPES numpy_type<npy_int8>() { return NPY_INT8; }
        template <>
        inline NPY_TYPES numpy_type<npy_int16>() { return NPY_INT16; }
        template <>
        inline NPY_TYPES numpy_type<npy_int32>() { return NPY_INT32; }
        template <>
        inline NPY_TYPES numpy_type<npy_int64>() { return NPY_INT64; }
//        template <>
//        inline NPY_TYPES numpy_type<npy_uint8>() { return NPY_UINT8; }
        template <>
        inline NPY_TYPES numpy_type<npy_uint16>() { return NPY_UINT16; }
        template <>
        inline NPY_TYPES numpy_type<npy_uint32>() { return NPY_UINT32; }
        template <>
        inline NPY_TYPES numpy_type<npy_uint64>() { return NPY_UINT64; }
//        template <>
//        inline NPY_TYPES numpy_type<npy_float16>() { return NPY_FLOAT16; }
        template <>
        inline NPY_TYPES numpy_type<npy_float32>() { return NPY_FLOAT32; }
        template <>
        inline NPY_TYPES numpy_type<npy_float64>() { return NPY_FLOAT64; }
        template <>
        inline NPY_TYPES numpy_type<npy_float128>() { return NPY_FLOAT128; }
//        template <>
//        inline NPY_TYPES numpy_type<double>() { return NPY_DOUBLE; }
//        template <>
//        inline NPY_TYPES numpy_type<float>() { return NPY_FLOAT; }

        template<typename T>
        inline PyObject * numpy_from_data(
                T* buffer,
                NPY_TYPES dtype,
                std::vector<size_t>& shape
        ) {
            _np_init();

            auto nd = shape.size();
            auto dims = (npy_intp*) shape.data();
//            py_printf("huh fack %lu %lu %lu %lu\n", dims[0], dims[1], dims[2], dims[3]);
            auto data = (void*)buffer;
            PyObject *dat = PyArray_SimpleNewFromData(
                    nd,
                    dims,
                    dtype,
                    data
            );

//            py_printf("huh fack\n");
            if (dat == NULL) {
                throw std::runtime_error("bad numpy shit");
            }
//            else {
//                py_printf("huh fack2...\n");
//                throw std::runtime_error("good numpy shit");
//            }

            return dat;
        }
        template<typename T>
        inline PyObject* numpy_from_data(
                T* buffer,
                std::vector<size_t> shape
        ) {
            auto shp = shape;
            auto npy_type = numpy_type<T>();
            return numpy_from_data<T>(buffer, npy_type, shp );
        }
        template<>
        inline PyObject* numpy_from_data<double>(
                double* buffer,
                std::vector<size_t> shape
        ) {
//            py_printf("....? %f %f %f\n", buffer[0], buffer[1], buffer[2]);
            auto wat = numpy_from_data<double>(buffer, NPY_FLOAT64, shape );
//            print_obj("wat %s\n", wat);
            return wat;
        }

        inline PyObject* numpy_copy_array(PyObject* obj) {
            _np_init();
            _check_py_arr(obj);
            auto arr = (PyArrayObject*) obj; // I think this is how one builds an array obj...?
            auto descr = PyArray_DESCR(arr);
            Py_XINCREF(descr);
            return PyArray_FromArray(arr, descr, NPY_ARRAY_ENSURECOPY);
        }
        template<typename T>
        inline PyObject* as_python(std::vector<T> data) {
            if (!data.empty()) {
                std::vector<size_t> shape = {data.size()};
                T *buffer = data.data();
                auto arr = numpy_from_data<T>(buffer, shape);
                return numpy_copy_array(arr); // since we pass by value we need to copy again...
            } else {
                npy_intp dims[0];
                auto npy_type = numpy_type<T>();
                auto arr = PyArray_EMPTY(0, dims, npy_type, false);
                return arr;
            }
        }
        template<>
        inline PyObject* as_python<bool>(std::vector<bool> data) {
            if (!data.empty()) {
                auto nels = data.size();
                std::vector<size_t> shape = {nels};
                bool buffer[nels];
                for (size_t i=0; i < nels; i++) { buffer[nels] = data[i]; }
                auto arr = numpy_from_data<bool>(buffer, shape);
                return numpy_copy_array(arr); // We'll let other parts of the code-base do copies if they want
            } else {
                npy_intp dims[0];
                auto npy_type = numpy_type<bool>();
                auto arr = PyArray_EMPTY(0, dims, npy_type, false);
                return arr;
            }
        }
        template<>
        inline PyObject* as_python<PyObject*>(std::vector<PyObject*> data) {
            return as_python_tuple<PyObject *>(data);
        }


        inline std::vector<size_t> numpy_shape_as_size_t(PyObject* obj) {
            _np_init();
            _check_py_arr(obj);
            auto arr = (PyArrayObject*) obj; // I think this is how one builds an array obj...?
            auto shp = (size_t*) PyArray_SHAPE(arr);
            return std::vector<size_t>(shp, shp + PyArray_NDIM(arr));
        }
        template <typename T>
        inline std::vector<T> numpy_shape(PyObject* obj) {
            auto base_shape = numpy_shape_as_size_t(obj);
            return std::vector<T>(base_shape.begin(), base_shape.end());
        }
        template <>
        inline std::vector<size_t> numpy_shape<size_t>(PyObject* obj) {
            return numpy_shape_as_size_t(obj);
        }

        inline std::string get_py_err_msg() {
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            std::string err_msg;
            if (ptype != NULL) {
                if (pvalue != NULL && ptype != PyExc_SystemError) { err_msg = get_python_repr(pvalue); }
                PyErr_Restore(ptype, pvalue, ptraceback);
            }
            return err_msg;
        }

        template <typename T>
        inline T get_python_attr(PyObject* obj, std::string& attr) {
            auto attr_ob = get_python_attr<PyObject*>(obj, attr);
            if (pyadeeb.debug_print()) {
                auto garb = get_python_repr(attr_ob);
                py_printf("    > got attr \"%s\", now converting %s to value\n", attr.c_str(), garb.c_str());
            }
            try {
                auto val = from_python<T>(attr_ob);
                Py_XDECREF(attr_ob); // annoying...
                return val;
            } catch (std::exception &e) {
                Py_XDECREF(attr_ob);
                throw;
            }
        }
        template<>
        inline PyObject* get_python_attr<PyObject *>(PyObject* obj, std::string& attr) {
            if (obj == NULL) {
                if (pyadeeb.debug_print()) py_printf("ERROR: object is NULL! (can't get %s)\n", attr.c_str());
                std::string err = "requested attrib " + attr + " from NULL object";
                PyErr_SetString(PyExc_TypeError, err.c_str());
                throw std::runtime_error("no obj issue");
            }
            PyObject* ret = PyObject_GetAttrString(obj, attr.c_str());
            if (ret == NULL) {
                if (pyadeeb.debug_print()) py_printf("ERROR: failed to get attr %s\n", attr.c_str());
                if (get_py_err_msg().empty()) {
                    std::string err = "Requested attrib. \"" + attr + "\" but got NULL back...?";
                    PyErr_SetString(PyExc_AttributeError, err.c_str());
                }
                throw std::runtime_error("attribute error");
            }
            return ret;
        }
        template <typename T>
        inline T get_python_attr(PyObject* obj, const char* attr) {
            std::string attr_str = attr;
            if (pyadeeb.debug_print()) {
                auto garb = get_python_repr(obj);
                py_printf("    > getting attr \"%s\" from %s\n", attr, garb.c_str());
            }
            return get_python_attr<T>(obj, attr_str);
        }

        template <typename T>
        inline std::vector<T> get_python_attr_iterable(PyObject* obj, std::string& attr) {
            auto attr_ob = get_python_attr<PyObject *>(obj, attr);
            if (pyadeeb.debug_print()) py_printf("  > getting iterable out of object attr %s\n", attr.c_str());
            try {
                auto val = from_python_iterable<T>(attr_ob);
                Py_XDECREF(attr_ob); // annoying...
                return val;
            } catch (std::exception &e) {
                Py_XDECREF(attr_ob);
                throw;
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
                throw;
            }
        }
        template <>
        inline PyObject* get_python_attr_ptr<PyObject>(PyObject* obj, std::string& attr) {
            return get_python_attr<PyObject *>(obj, attr);
        }
        template <typename T>
        inline T* get_python_attr_ptr(PyObject* obj, const char* attr) {
            std::string attr_str = attr;
            return get_python_attr_ptr<T>(obj, attr_str);
        }

    }

}

#endif //RYNLIB_PYALLUP_HPP
