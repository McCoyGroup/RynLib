#include "RynLib.h"

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

int int3d(int i, int j, int k, int m, int l) {
    return (m*l) * i + (l*j) + k;
}

std::vector<std::string> _getAtomTypes( PyObject* atoms, Py_ssize_t num_atoms ) {

    std::vector<std::string> mattsAtoms(num_atoms);
    for (int i = 0; i<num_atoms; i++) {
        PyObject* atom = PyList_GetItem(atoms, i);
        PyObject* pyStr = NULL;
        const char* atomStr = _GetPyString(atom, pyStr);
        std::string atomString = atomStr;
        mattsAtoms[i] = atomString;
        Py_XDECREF(atom);
        Py_XDECREF(pyStr);
    }

    return mattsAtoms;
}

std::vector< std::vector<double> > _getWalkerCoords(double* raw_data, int i, Py_ssize_t num_atoms) {
    std::vector< std::vector<double> > walker_coords(num_atoms, std::vector<double>(3));
    for (int j = 0; j<num_atoms; j++) {
        for (int k = 0; k<3; k++) {
            walker_coords[j][k] = raw_data[int3d(i, j, k, num_atoms, 3)];
        }
    };
    return walker_coords;
}

PyObject *RynLib_callPot( PyObject* args, PyObject* kwargs ) {

    PyObject* atoms;
    PyObject* coords;
    if ( !PyArg_ParseTuple(args, "OO", &atoms, &coords) ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    std::vector<std::string> mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    double* raw_data = _GetDoubleDataArray(coords);
    std::vector< std::vector<double> > walker_coords = _getWalkerCoords(raw_data, 0, num_atoms);
    double pot = MillerGroup_entosPotential(walker_coords, mattsAtoms);

    PyObject *potVal = Py_BuildValue("f", pot);
    return potVal;

}

PyObject *RynLib_callPotVec( PyObject* args, PyObject* kwargs ) {
    // vector version of callPot

    PyObject* atoms;
    PyObject* coords;
    if ( !PyArg_ParseTuple(args, "OO", &atoms, &coords) ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    std::vector<std::string> mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    Py_ssize_t num_walkers = PyObject_Length(atoms);
    std::vector<double> potVals(num_walkers);
    double* raw_data = _GetDoubleDataArray(coords);
    for (int i = 0; i<num_walkers; i++) {
        std::vector< std::vector<double> > walker_coords = _getWalkerCoords(raw_data, i, num_atoms);
        double pot = MillerGroup_entosPotential(walker_coords, mattsAtoms);
        potVals[i] = pot;
    };

    // TODO: NEED TO CONVERT BACK TO A NUMPY ARRAY IF WE DECIDE TO USE THIS BRANCH OF THE CODE
    Py_RETURN_NONE;

}

PyObject *RynLib_testPot( PyObject* args, PyObject* kwargs ) {

    PyObject *hello;

    hello = Py_BuildValue("f", 50.2);
    return hello;

}

static PyMethodDef RynLibMethods[] = {
    {"rynaLovesDMC", RynLib_callPot, METH_VARARGS, "calls entos on a single walker"},
    {"rynaLovesDMCLots", RynLib_callPotVec, METH_VARARGS, "will someday call entos on a vector of walkers"},
    {"rynaSaysYo", RynLib_testPot, METH_VARARGS, "a test flat potential for debugging"}
};


#if (PY_MAJOR_VERSION == 3)
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
#else
PyMODINIT_FUNC initRynLib(void)
{
    (void) Py_InitModule("RynLib", RynLibMethods);
}
#endif
