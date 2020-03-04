#include "Dumpi.h"
#include "RynTypes.hpp"

void _mpiInit(int* world_size, int* world_rank) {
    *world_size = 1;
    *world_rank = 0;
}

void _mpiFinalize() {
    // boop
}

void _mpiBarrier() {
    // boop
}

void _mpiBindComm(PyObject *cls) {
    PyObject_SetAttrString(cls, "_comm",
            PyCapsule_New((void *)MPI_COMM_WORLD, "_COMM_WORLD", NULL);)
    PyObject_SetAttrString(cls, "_scatter_walkers",
            PyCapsule_New((void *)Scatter_walkers, "_SCATTER_WALKERS", NULL););
    PyObject_SetAttrString(cls, "_gather_walkers",
            PyCapsule_New((void *)Gather_walkers, "_GATHER_WALKERS", NULL););
};

int Scatter_Walkers(
        PyObject *manager,
        RawWalkerBuffer raw_data,
        int walkers_to_core, int walker_cnum,
        RawWalkerBuffer walker_buf
        ) {
    PyObject *comm_capsule = PyObject_GetAttrString(manager, "comm");
    MPI_Comm comm = (MPI_Comm) PyCapsule_GetPointer(comm_capsule, "_COMM_WORLD");
    return MPI_Scatter(
            raw_data,  // raw data buffer to chunk up
            walkers_to_core * walker_cnum, // three coordinates per atom per num_atoms per walker
            MPI_DOUBLE, // coordinates stored as doubles
            walker_buf, // raw array to write into
            walkers_to_core * walker_cnum, // three coordinates per atom per num_atoms per walker
            MPI_DOUBLE, // coordinates stored as doubles
            0, // root caller
            comm // communicator handle
    );
}

int Gather_Walkers(
        PyObject *manager,
        PotentialVector pots,
        int walkers_to_core,
        RawPotentialBuffer pot_buf
) {
    PyObject *comm_capsule = PyObject_GetAttrString(manager, "comm");
    MPI_Comm comm = (MPI_Comm) PyCapsule_GetPointer(comm_capsule, "_COMM_WORLD");
    return MPI_Gather(
            pots.data(),
            walkers_to_core, // number of walkers fed in
            MPI_DOUBLE, // coordinates stored as doubles
            pot_buf, // buffer to get the potential values back
            walkers_to_core, // number of walkers fed in
            MPI_DOUBLE, // coordinates stored as doubles
            0, // where they should go
            comm // communicator handle
    );
}


// MPI COMMUNICATION METHODS
PyObject *Dumpi_initializeMPI(PyObject *self, PyObject *args) {

    PyObject *hello, *cls;
    int world_size, world_rank;
    if ( !PyArg_ParseTuple(args, "O", &cls) ) return NULL;
    _mpiInit(&world_size, &world_rank);
    _mpiBindComm(cls);
    hello = Py_BuildValue("(ii)", world_rank, world_size);
    return hello;
}

PyObject *Dumpi_finalizeMPI(PyObject *self, PyObject *args) {
    _mpiFinalize();
    Py_RETURN_NONE;
}

PyObject *Dumpi_holdMyPI( PyObject* self, PyObject* args ) {
    _mpiBarrier();
    Py_RETURN_NONE;
}

// PYTHON WRAPPER EXPORT

static PyMethodDef DumpiMethods[] = {
    {"giveMePI", Dumpi_initializeMPI, METH_VARARGS, "calls Init and returns the processor rank"},
    {"noMorePI", Dumpi_finalizeMPI, METH_VARARGS, "calls Finalize in a safe fashion (can be done more than once)"},
    {"holdMyPI", Dumpi_holdMyPI, METH_VARARGS, "calls Barrier"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION > 2

const char Dumpi_doc[] = "Dumpi is for a dumpy interface to MPI";
static struct PyModuleDef DumpiModule = {
    PyModuleDef_HEAD_INIT,
    "Dumpi",   /* name of module */
    Dumpi_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    DumpiMethods
};

PyMODINIT_FUNC PyInit_Dumpi(void)
{
    return PyModule_Create(&DumpiModule);
}
#else

PyMODINIT_FUNC initDumpi(void)
{
    (void) Py_InitModule("Dumpi", DumpiMethods);
}

#endif

#endif