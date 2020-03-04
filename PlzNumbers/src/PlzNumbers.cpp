#include "PlzNumbers.h"

#include "Potators.hpp"
#include "PyAllUp.hpp"
#include "Flargs.hpp"
#include <stdexcept>

PyObject *PlzNumbers_callPot(PyObject* self, PyObject* args ) {

    PyObject* atoms;
    PyObject* coords;
    PyObject* pot_function;
    const char* bad_walkers_file;
    double err_val;
    bool raw_array_pot;
    if ( !PyArg_ParseTuple(args, "OOOOdp", &atoms, &coords, &pot_function, &bad_walkers_file, &err_val, &raw_array_pot) ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;
    Coordinates walker_coords = _getWalkerCoords(raw_data, 0, num_atoms);

    PotentialFunction PyCapsule_

    double pot;
    pot = _doopAPot(
            walker_coords,
            mattsAtoms,
            pot_function,
            bad_walkers_file,
            err_val,
            false
            );

    PyObject *potVal = Py_BuildValue("f", pot);
    return potVal;

}

PyObject *PlzNumbers_callPotVec( PyObject* self, PyObject* args ) {
    // vector version of callPot

    PyObject* atoms;
    PyObject* coords;
    PyObject* pot_function;
    PyObject* bad_walkers_file;
    double err_val;
    bool raw_array_pot, vectorized_potential;
    PyObject* manager;
    if ( !PyArg_ParseTuple(args, "OOOOdppO",
            &atoms,
            &coords,
            &pot_function,
            &bad_walkers_file,
            &err_val,
            &raw_array_pot,
            &vectorized_potential,
            &manager
            ) ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    if (PyErr_Occurred()) return NULL;
    Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // we'll assume we have number of walkers X ncalls X number of atoms X 3
    PyObject *shape = PyObject_GetAttrString(coords, "shape");
    if (shape == NULL) return NULL;

    PyObject *num_walkers_obj = PyTuple_GetItem(shape, 0);
    if (num_walkers_obj == NULL) return NULL;
    Py_ssize_t num_walkers = _FromInt(num_walkers_obj);
    if (PyErr_Occurred()) return NULL;

    PyObject *ncalls_obj = PyTuple_GetItem(shape, 1);
    if (ncalls_obj == NULL) return NULL;
    Py_ssize_t ncalls = _FromInt(ncalls_obj);
    if (PyErr_Occurred()) return NULL;

    // this thing should have the walker number as the slowest moving index then the number of the timestep
    // that way we'll really have the correct memory entering into our calls
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;

    // We can tell if MPI is active or not by whether COMM is None or not
    PotentialArray pot_vals;
    if (manager == Py_None) {
        pot_vals = _noMPIGetPot(
                raw_data,
                mattsAtoms,
                ncalls,
                num_walkers,
                num_atoms,
                pot_function,
                bad_walkers_file,
                err_val,
                vectorized_potential
                )
    } else {
        pot_vals = _mpiGetPot(
                manager,
                raw_data,
                mattsAtoms,
                ncalls,
                num_walkers,
                num_atoms,
                pot_function,
                bad_walkers_file,
                err_val,
                vectorized_potential
        );
    }

    bool main_core = true;
    if ( manager != Py_None ){
        PyObject *rank = PyObject_GetAttrString(manager, "world_rank");
        if (rank == NULL) { return NULL; }
        int rank = _FromInt(rank);
        Py_XDECREF(rank)
    }
    if ( main_core ){
        return _fillNumPyArray(pot_vals, num_walkers, ncalls);
    } else {
        Py_RETURN_NONE;
    }

}

// PYTHON WRAPPER EXPORT

static PyMethodDef PlzNumbersMethods[] = {
    {"rynaLovesPoots", PlzNumbers_callPot, METH_VARARGS, "calls a potential on a single walker"},
    {"rynaLovesPootsLots", PlzNumbers_callPotVec, METH_VARARGS, "calls a potential on a vector of walkers"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION > 2

const char PlzNumbers_doc[] = "PlzNumbers manages the calling of a potential at the C++ level";
static struct PyModuleDef PlzNumbersModule = {
    PyModuleDef_HEAD_INIT,
    "PlzNumbers",   /* name of module */
    PlzNumbers_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    PlzNumbersMethods
};

PyMODINIT_FUNC PyInit_PlzNumbers(void)
{
    return PyModule_Create(&PlzNumbersModule);
}
#else

PyMODINIT_FUNC initPlzNumbers(void)
{
    (void) Py_InitModule("PlzNumbers", PlzNumbersMethods);
}

#endif