#include "RynLib.h"

const char *_GetPyString( PyObject* s, const char *enc, const char *err, PyObject *pyStr) {
    pyStr = PyUnicode_AsEncodedString(s, enc, err);
    if (pyStr == NULL) return NULL;
    const char *strExcType =  PyBytes_AsString(pyStr);
//    Py_XDECREF(pyStr);
    return strExcType;
}

Py_buffer _GetDataBuffer(PyObject *data) {
    Py_buffer view;
    PyObject_GetBuffer(data, &view, PyBUF_CONTIG_RO);
    return view;
}

double *_GetDoubleDataBufferArray(Py_buffer *view) {
    double *c_data;
    if ( view == NULL ) return NULL;
    c_data = (T *) view->buf;
    if (c_data == NULL) {
        PyBuffer_Release(view);
    }
    return c_data;
}

double *_GetDoubleDataArray(PyObject *data) {
    Py_buffer view = _GetDataBuffer(data);
    double *array = _GetDoubleDataBufferArray<T>(&view);
//    CHECKNULL(array);
    return array;
}

int ind3d(int i, int j, int k, int n, int m, int l) {
    return (m*l) * i + (l*j) + k;
}

RynLib_callPot( PyObject* args ) {

    PyObject* atoms;
    PyObject* coords;
    Py_ParseArgs("OO", &atoms, &coords);

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    std::vector<std::string> mattsAtoms(num_atoms);
    for (int i = 0; i<num_atoms; i++) {
        PyObject* atom = PyIter_Next(iterator);
        const char* atomStr = _GetPyString(atom);
        std::string atomString = atomStr;
        mattsAtoms[i] = atomString;
        Py_XDECREF(atom);
    }

    // Assumes number of walkers X number of atoms X 3
    Py_ssize_t num_walkers = PyObject_Length(atoms);
    std::vector< std::vector< std::vector<double> > > mattsCoords(num_walkers, num_atoms, 3);
    double* raw_data = _GetDoubleDataArray<double>(coords);
    for (int i = 0; i<num_walkers; i++) {
        std::vector< std::vector<double> > walker_coords(num_atoms, 3);
        for (int j = 0; i<num_atoms; i++) {
            for (int k = 0; k<3; k++) {
                walker_coords[j][k] = raw_data[int3d(i, j, k, num_walkers, num_atoms, 3)];
            }
        };
        mattsCoords[n] = walker_coords;
    };

    MillerGroup_entosPotential(mattsAtoms, mattsCoords);

    Py_RETURN_NONE;

}

static PyMethodDef RynLibMethods[] = {
    {"rynaLovesDMC", RynLib_callPot, METH_VARARGS, ""}
};

const char RynLib_doc[] = "RynLib is for Ryna Dorisii";

static struct PyModuleDef RynLibModule = {
    PyModuleDef_HEAD_INIT,
    "RynLib",   /* name of module */
    RynLib_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    RynLibMethods
};

PyMODINIT_FUNC PyInit_RynLib(void)
{
    return PyModule_Create(&RynLibModule);
}
