//
// Created by Mark Boyer on 1/30/20.
//

#include "Python.h"
#include "RynTypes.hpp"

#if PY_MAJOR_VERSION == 3

const char *_GetPyString( PyObject* s, const char *enc, const char *err, PyObject *pyStr) {
    pyStr = PyUnicode_AsEncodedString(s, enc, err);
    if (pyStr == NULL) return NULL;
    const char *strExcType =  PyBytes_AsString(pyStr);
//    Py_XDECREF(pyStr);
    return strExcType;
}
const char *_GetPyString( PyObject* s, PyObject *pyStr) {
    // unfortunately we need to pass the second pyStr so we can XDECREF it later
    return _GetPyString( s, "utf-8", "strict", pyStr); // utf-8 is safe since it contains ASCII fully
    }

Py_ssize_t _FromInt( PyObject* int_obj ) {
    return PyLong_AsSsize_t(int_obj);
}

bool _FromBool( PyObject* bool_obj ) {
    return PyObject_IsTrue(bool_obj);
}

Real_t _FromFloat(PyObject* float_obj) {
    return PyFloat_AsDouble(float_obj);
}

#else
const char *_GetPyString( PyObject* s ) {
    return PyString_AsString(s);
}
const char *_GetPyString( PyObject* s, PyObject *pyStr ) {
    // just to unify the 2/3 interface
    return _GetPyString( s );
}
Py_ssize_t _FromInt( PyObject* int_obj ) {
    return PyInt_AsSsize_t(int_obj);
}

bool _FromBool( PyObject* bool_obj ) {
    return PyObject_IsTrue(bool_obj);
}

Real_t _FromFloat(PyObject* float_obj) {
    return PyFloat_AsDouble(float_obj);
}

#endif

Names _getAtomTypes( PyObject* atoms, Py_ssize_t num_atoms ) {

    Names mattsAtoms(num_atoms);
    for (int i = 0; i<num_atoms; i++) {
        PyObject* atom = PyList_GetItem(atoms, i);
        PyObject* pyStr = NULL;
        const char* atomStr = _GetPyString(atom, pyStr);
        Name atomString = atomStr;
        mattsAtoms[i] = atomString;
//        Py_XDECREF(atom);
        Py_XDECREF(pyStr);
    }

    return mattsAtoms;
}

Py_buffer _GetDataBuffer(PyObject *data) {
    Py_buffer view;
    PyObject_GetBuffer(data, &view, PyBUF_CONTIG_RO);
    return view;
}

double *_GetDoubleDataBufferArray(Py_buffer *view) {
    double *c_data;
    if ( view == NULL ) return NULL;
    c_data = (double *) view->buf;
    if (c_data == NULL) {
        PyBuffer_Release(view);
    }
    return c_data;
}

double *_GetDoubleDataArray(PyObject *data) {
    Py_buffer view = _GetDataBuffer(data);
    double *array = _GetDoubleDataBufferArray(&view);
//    CHECKNULL(array);
    return array;
}

PyObject *_getNumPyZerosMethod() {
    PyObject *array_module = PyImport_ImportModule("numpy");
    if (array_module == NULL) return NULL;
    PyObject *builder = PyObject_GetAttrString(array_module, "zeros");
    Py_XDECREF(array_module);
    if (builder == NULL) return NULL;
    return builder;
};

PyObject *_getNumPyArray(
        int n,
        int m,
        const char *dtype
) {
    // Initialize NumPy array of correct size and dtype
    PyObject *builder = _getNumPyZerosMethod();
    if (builder == NULL) return NULL;
    PyObject *dims = Py_BuildValue("((ii))", n, m);
    Py_XDECREF(builder);
    if (dims == NULL) return NULL;
    PyObject *kw = Py_BuildValue("{s:s}", "dtype", dtype);
    if (kw == NULL) return NULL;
    PyObject *pot = PyObject_Call(builder, dims, kw);
    Py_XDECREF(kw);
    Py_XDECREF(dims);
    return pot;
}

PyObject *_getNumPyArray(
        int n,
        int m,
        int l,
        const char *dtype
) {
    // Initialize NumPy array of correct size and dtype
    PyObject *builder = _getNumPyZerosMethod();
    if (builder == NULL) return NULL;
    PyObject *dims = Py_BuildValue("((iii))", n, m, l);
    Py_XDECREF(builder);
    if (dims == NULL) return NULL;
    PyObject *kw = Py_BuildValue("{s:s}", "dtype", dtype);
    if (kw == NULL) return NULL;
    PyObject *pot = PyObject_Call(builder, dims, kw);
    Py_XDECREF(kw);
    Py_XDECREF(dims);
    return pot;
}

PyObject *_getNumPyArray(
        int n,
        int m,
        int l,
        int k,
        const char *dtype
) {
    // Initialize NumPy array of correct size and dtype
    PyObject *builder = _getNumPyZerosMethod();
    if (builder == NULL) return NULL;
    PyObject *dims = Py_BuildValue("((iiii))", n, m, l, k);
    Py_XDECREF(builder);
    if (dims == NULL) return NULL;
    PyObject *kw = Py_BuildValue("{s:s}", "dtype", dtype);
    if (kw == NULL) return NULL;
    PyObject *pot = PyObject_Call(builder, dims, kw);
    Py_XDECREF(kw);
    Py_XDECREF(dims);
    return pot;
}

// NumPy Communication Methods
PyObject *_fillNumPyArray(
        const PotentialArray &pot_vals,
        const int ncalls,
        const int num_walkers
) {

    // Initialize NumPy array of correct size and dtype
    PyObject *pot = _getNumPyArray(ncalls, num_walkers, "float");
    if (pot == NULL) return NULL;
    double *data = _GetDoubleDataArray(pot);
    for (int i = 0; i < ncalls; i++) {
        memcpy(
                // where in the data array memory to start copying to
                data + num_walkers * i,
                // where in the potential array to start copying from
                pot_vals[i].data(),
                // what
                sizeof(double) * num_walkers
        );
    };
    return pot;
}

PyObject *_fillNumPyArray(
        RawPotentialBuffer pot_vals,
        const int ncalls,
        const int num_walkers
) {
    // Initialize NumPy array of correct size and dtype
    PyObject *pot = _getNumPyArray(ncalls, num_walkers, "float");
    if (pot == NULL) return NULL;
    double *data = _GetDoubleDataArray(pot);
    memcpy(data, pot_vals, sizeof(double)  * num_walkers * num_walkers);
    return pot;
}

PyObject *_fillWalkersNumPyArray(
        const RawWalkerBuffer coords,
        const int num_walkers,
        const int natoms
) {
    // Initialize NumPy array of correct size and dtype
    PyObject *walkers = _getNumPyArray(num_walkers, natoms, 3, "float");
    if (walkers == NULL) return NULL;
    double *data = _GetDoubleDataArray(walkers);
    memcpy(data, coords, sizeof(double)  * num_walkers * natoms * 3);
    return walkers;
}

void _printObject(const char* fmtString, PyObject *obj) {
    PyObject *str=NULL;
    PyObject *repr=PyObject_Repr(obj);
    printf(fmtString, _GetPyString(repr, str));
    Py_XDECREF(repr);
    Py_XDECREF(str);
}