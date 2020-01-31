#include "RynLib.h"

#ifdef SADBOYDEBUG

static PyMethodDef RynLibMethods[] = {
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initRynLib(void)
{
    (void) Py_InitModule("RynLib", RynLibMethods);
}

#else

#include "Potators.hpp"
#include "PyAllUp.hpp"
#include "Flargs.hpp"
#include <stdexcept>

#ifdef I_HAVE_PIE

#include "MyPieIsHere.hpp"

#else

#include "MyPieIsGone.hpp"

#endif


PyObject *RynLib_callPot(PyObject* self, PyObject* args ) {

    PyObject* atoms;
    PyObject* coords;
    if ( !PyArg_ParseTuple(args, "OO", &atoms, &coords) ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;
    Coordinates walker_coords = _getWalkerCoords(raw_data, 0, num_atoms);
    double pot;
//    try {
    pot = _doopAPot(walker_coords, mattsAtoms);
//    } catch(...) {
//        return NULL;
//    }

    PyObject *potVal = Py_BuildValue("f", pot);
    return potVal;

}

PyObject *RynLib_callPotVec( PyObject* self, PyObject* args ) {
    // vector version of callPot

    PyObject* atoms;
    PyObject* coords;
    if ( !PyArg_ParseTuple(args, "OO", &atoms, &coords) ) return NULL;

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

    int world_size, world_rank;
    _mpiInit(&world_size, &world_rank);
    // Get vector of values from MPI call
    PotentialArray pot_vals = _mpiGetPot(
                raw_data, mattsAtoms,
                ncalls,
                num_walkers,
                num_atoms,
                world_size,
                world_rank
                );

    if ( world_rank == 0 ){
        return _fillNumPyArray(pot_vals, num_walkers, ncalls);
    } else {
        Py_RETURN_NONE;
    }

}

PyObject *RynLib_testPot( PyObject* self, PyObject* args ) {

    PyObject *hello;

    hello = Py_BuildValue("f", 50.2);
    return hello;

}


// MPI COMMUNICATION METHODS
PyObject *RynLib_initializeMPI(PyObject *self, PyObject *args) {

    PyObject *hello;
    int world_size, world_rank;
    _mpiInit(&world_size, &world_rank);
    hello = Py_BuildValue("(ii)", world_rank, world_size);
    return hello;

}

PyObject *RynLib_finalizeMPI(PyObject *self, PyObject *args) {

    _mpiFinalize();
    Py_RETURN_NONE;

}

PyObject *RynLib_holdMyPI( PyObject* self, PyObject* args ) {

    _mpiBarrier();
    Py_RETURN_NONE;

}

#ifdef AGE_OF_AQUARIUS

#include "WalkerPropagator.h"

// New design that will make use of the WalkerPropagator object I'm setting up
PyObject *RynLib_getWalkers( PyObject* self, PyObject* args ) {

    PyObject* cores;
    if ( !PyArg_ParseTuple(args, "O", &cores) ) return NULL;

    Coordinates walker_positions = _mpiGetWalkersFromNodes(

    );

}

#else

PyObject *RynLib_getWalkers( PyObject* self, PyObject* args ) {
    Py_RETURN_NOTIMPLEMENTED;
}

#endif


PyObject *RynLib_setBadWalkerFile( PyObject* self, PyObject* args ) {

    PyObject* out;
    if ( !PyArg_ParseTuple(args, "O", &out) ) return NULL;
    PyObject* pyStr = NULL;
    const char* fileFlarg = _GetPyString(out, pyStr);
    Py_XDECREF(pyStr);

    BAD_WALKERS_WHATCHA_GONNA_DO = fileFlarg;

    Py_RETURN_NONE;

}

PyObject *RynLib_setOnlyHF( PyObject* self, PyObject* args ) {

    bool* useHF;
    if ( !PyArg_ParseTuple(args, "b", &useHF) ) return NULL;

    MACHINE_LERNING_IS_A_SCAM = useHF;

    Py_RETURN_NONE;

}

PyObject *RynLib_setPotential( PyObject* self, PyObject* args ) {

    PyObject* potCap;
    PyObject* capName;
    if ( !PyArg_ParseTuple(args, "OO", &potCap, &capName) ) return NULL;

    PotentialFunction pot_pointer;
    if ( capName == Py_None ) {
        pot_pointer = (PotentialFunction) PyCapsule_GetPointer(potCap, NULL);
    } else {
        PyObject* pyStr = NULL;
        const char* capNameStr = _GetPyString(capName, pyStr);
        pot_pointer = (PotentialFunction) PyCapsule_GetPointer(potCap, capNameStr);
        Py_XDECREF(pyStr);
    }

    POOTY_PATOOTY = pot_pointer;

    Py_RETURN_NONE;

}

// PYTHON WRAPPER EXPORT

static PyMethodDef RynLibMethods[] = {
    {"rynaLovesDMC", RynLib_callPot, METH_VARARGS, "calls entos on a single walker"},
    {"rynaLovesDMCLots", RynLib_callPotVec, METH_VARARGS, "calls entos on a vector of walkers"},
    {"rynaSaysYo", RynLib_testPot, METH_VARARGS, "a test flat potential for debugging"},
    {"giveMePI", RynLib_initializeMPI, METH_VARARGS, "calls Init and returns the processor rank"},
    {"noMorePI", RynLib_finalizeMPI, METH_VARARGS, "calls Finalize in a safe fashion (can be done more than once)"},
    {"holdMyPI", RynLib_holdMyPI, METH_VARARGS, "calls Barrier"},
    {"doITrustThemachines", RynLib_setBadWalkerFile, METH_VARARGS, "sets the flag to use HF only or not"},
    {"whatToDoWithAProblemLikeRyna", RynLib_setBadWalkerFile, METH_VARARGS, "sets the bad walker dump file"},
    {"walkyTalky", RynLib_getWalkers, METH_VARARGS, "gets Walkers in a WalkerPropagator env"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION > 2

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

#endif