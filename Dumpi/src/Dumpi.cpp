#include "Dumpi.h"
#include "RynTypes.hpp"

void _mpiInit(int* world_size, int* world_rank) {
    // Initialize MPI state
    int did_i_do_good_pops = 0;
    int err = MPI_SUCCESS;
    MPI_Initialized(&did_i_do_good_pops);
    if (!did_i_do_good_pops){
//        int error;
//        error = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        printf("this is before MPI_Init\n");
//        MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        MPI_ERRORS_ARE_FATAL = MPI_ERRORS_RETURN; // eesh this is dangerous, but I just want to see if it works...
        err = MPI_Init(NULL, NULL);
        printf("...okay I guess MPI initialized\n");
//        error = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    };
    if (err == MPI_SUCCESS) {
        MPI_Comm_size(MPI_COMM_WORLD, world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, world_rank);
    }
    // printf("This many things %d and I am here %d\n", *world_size, *world_rank);
}

void _mpiFinalize() {
    int did_i_do_bad_pops = 0;
    MPI_Finalized(&did_i_do_bad_pops); // need to check if we called Init once already
    if (!did_i_do_bad_pops){
        MPI_Finalize();
    };
}

void _mpiBarrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

static int Scatter_Walkers(
        PyObject *manager,
        RawWalkerBuffer raw_data,
        int walkers_to_core, int walker_cnum,
        RawWalkerBuffer walker_buf
        ) {
    PyObject *comm_capsule = PyObject_GetAttrString(manager, "comm");
    if (comm_capsule == NULL) {
        return -1;
    }
    MPI_Comm comm = (MPI_Comm) PyCapsule_GetPointer(comm_capsule, "Dumpi._COMM_WORLD");
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

static int Gather_Walkers(
        PyObject *manager,
        RawPotentialBuffer pots,
        int walkers_to_core,
        RawPotentialBuffer pot_buf
) {
//    printf("trying to get comm...\n");
    PyObject *comm_capsule = PyObject_GetAttrString(manager, "comm");
    if (comm_capsule == NULL) {
        return -1;
    }
    MPI_Comm comm = (MPI_Comm) PyCapsule_GetPointer(comm_capsule, "Dumpi._COMM_WORLD");
//    printf("got COMM so now gathering %d walkers\n", walkers_to_core);
    return MPI_Gather(
            pots,
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

    PyObject *hello;//, *cls;
    int world_size, world_rank;
    world_size = -1;
    world_rank = -1;
//    if ( !PyArg_ParseTuple(args, "O", &cls) ) return NULL;
    _mpiInit(&world_size, &world_rank); // If this fails, nothing is set
    if (world_rank == -1) {
        hello = NULL;
        PyErr_SetString(PyExc_IOError, "MPI failed to initialize");
    } else {
        hello = Py_BuildValue("(ii)", world_rank, world_size);
        };
    return hello;
}

PyObject *Dumpi_finalizeMPI(PyObject *self, PyObject *args) {
    _mpiFinalize();
    Py_RETURN_NONE;
}

PyObject *Dumpi_syncMPI( PyObject* self, PyObject* args ) {
    _mpiBarrier();
    Py_RETURN_NONE;
}

PyObject *Dumpi_abortMPI( PyObject* self, PyObject* args ) {
    MPI_Abort(MPI_COMM_WORLD, 303);
    Py_RETURN_NONE;
}

// PYTHON WRAPPER EXPORT

static PyMethodDef DumpiMethods[] = {
    {"giveMePI", Dumpi_initializeMPI, METH_VARARGS, "calls Init and returns the processor rank"},
    {"noMorePI", Dumpi_finalizeMPI, METH_VARARGS, "calls Finalize in a safe fashion (can be done more than once)"},
    {"holdMyPI", Dumpi_syncMPI, METH_VARARGS, "calls Barrier"},
    {"killMyPI", Dumpi_abortMPI, METH_VARARGS, "calls Abort"},
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
    PyObject* module = PyModule_Create(&DumpiModule);
    if (module == NULL) { return NULL; }

    static PyObject *comm_cap = PyCapsule_New((void *)MPI_COMM_WORLD, "Dumpi._COMM_WORLD", NULL);
    if (comm_cap == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Failed to create COMM_WORLD pointer capsule");
        return NULL;
    }
    if (PyModule_AddObject(module, "_COMM_WORLD", comm_cap) < 0) {
        Py_XDECREF(comm_cap);
        Py_DECREF(module);
        return NULL;
    }

    static PyObject *scatter_cap = PyCapsule_New((void *)Scatter_Walkers, "Dumpi._SCATTER_WALKERS", NULL);
    if (scatter_cap == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Failed to create Scatter pointer capsule");
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddObject(module, "_SCATTER_WALKERS", scatter_cap) < 0) {
        Py_XDECREF(scatter_cap);
        Py_DECREF(module);
        return NULL;
    }
    ScatterFunction test = (ScatterFunction) PyCapsule_GetPointer(scatter_cap, "Dumpi._SCATTER_WALKERS");
    if (test == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Scatter pointer is NULL");
        Py_XDECREF(scatter_cap);
        Py_DECREF(module);
        return NULL;
    }

    static PyObject *gather_cap = PyCapsule_New((void *)Gather_Walkers, "Dumpi._GATHER_WALKERS", NULL);
    if (gather_cap == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Failed to create Gather pointer capsule");
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddObject(module, "_GATHER_WALKERS", gather_cap) < 0) {
        Py_XDECREF(gather_cap);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
#else

PyMODINIT_FUNC initDumpi(void)
{
    (void) Py_InitModule("Dumpi", DumpiMethods);
}

#endif