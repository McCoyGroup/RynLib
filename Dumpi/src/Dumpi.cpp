#include "Dumpi.h"
#include "RynTypes.hpp"
#include <string>

static int _world_rank, _world_size;
void _mpiInit(int* world_size, int* world_rank) {
    // Initialize MPI state
    int did_i_do_good_pops = 0;
    int err = MPI_SUCCESS;
    MPI_Initialized(&did_i_do_good_pops);
    if (!did_i_do_good_pops){
        err = MPI_Init(NULL, NULL);
    };
    if (err == MPI_SUCCESS) {
        MPI_Comm_size(MPI_COMM_WORLD, world_size);
        _world_size = *world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, world_rank);
        _world_rank = *world_rank;
    }
}

int _mpiHandleErrors(int res) {
    int succeeded = 0;
    std::string base_message = "No error";
    switch (res) {
        case MPI_SUCCESS:
            succeeded = 1;
        case MPI_ERR_COMM:
            base_message = "Invalid communicator. A common error is to use a null communicator in a call (not even allowed in MPI_Comm_rank).";
        case MPI_ERR_COUNT:
            base_message = "Invalid count argument. Count arguments must be non-negative; a count of zero is often valid.";
        case MPI_ERR_TYPE:
            base_message = "Invalid datatype argument. Additionally, this error can occur if an uncommitted MPI_Datatype (see MPI_Type_commit) is used in a communication call.";
        case MPI_ERR_BUFFER:
            base_message = "Invalid buffer pointer. Usually a null buffer where one is not valid.";
        default:
            base_message = "Unknown MPI error (i.e. I didn't bake it in).";
    }

    if (!succeeded) {
//        const char* format = "Error code %d on %d: %s";
        std::string err = "Error code " + std::to_string(res) +
                " on " + std::to_string(_world_rank) + ": " +
                base_message;
        PyErr_SetString(PyExc_IOError, err.c_str());
    }

    return succeeded;

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
//    PyObject *comm_capsule = PyObject_GetAttrString(manager, "comm");
//    if (comm_capsule == NULL) {
//        return -1;
//    }

    int res =  MPI_Scatter(
            raw_data,  // raw data buffer to chunk up
            walkers_to_core * walker_cnum, // three coordinates per atom per num_atoms per walker
            MPI_DOUBLE, // coordinates stored as doubles
            walker_buf, // raw array to write into
            walkers_to_core * walker_cnum, // three coordinates per atom per num_atoms per walker
            MPI_DOUBLE, // coordinates stored as doubles
            0, // root caller
            MPI_COMM_WORLD // communicator handle
    );

    return _mpiHandleErrors(res);

}

static int Gather_Potentials(
        PyObject *manager,
        RawPotentialBuffer pots,
        int walkers_to_core,
        RawPotentialBuffer pot_buf
) {
//    PyObject *comm_capsule = PyObject_GetAttrString(manager, "comm");
//    if (comm_capsule == NULL) {
//        return -1;
//    }
//    MPI_Comm comm = (MPI_Comm) PyCapsule_GetPointer(comm_capsule, "Dumpi._COMM_WORLD");
//    printf("got COMM so now gathering %d walkers\n", walkers_to_core);
    int res = MPI_Gather(
            pots,
            walkers_to_core, // number of walkers fed in
            MPI_DOUBLE, // coordinates stored as doubles
            pot_buf, // buffer to get the potential values back
            walkers_to_core, // number of walkers fed in
            MPI_DOUBLE, // coordinates stored as doubles
            0, // where they should go
            MPI_COMM_WORLD // communicator handle
    );

    return _mpiHandleErrors(res);
}

static int Gather_Walkers(
        PyObject *manager,
        RawWalkerBuffer walkers,
        int walkers_to_core, int walker_cnum,
        RawWalkerBuffer walk_buf
) {
    int res = MPI_Gather(
            walkers,
            walkers_to_core * walker_cnum, // number of walkers fed in
            MPI_DOUBLE, // coordinates stored as doubles
            walk_buf, // buffer to get the walker values back
            walkers_to_core * walker_cnum, // number of walkers fed in
            MPI_DOUBLE, // coordinates stored as doubles
            0, // where they should go
            MPI_COMM_WORLD // communicator handle
    );

    return _mpiHandleErrors(res);
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
//    ScatterFunction test = (ScatterFunction) PyCapsule_GetPointer(scatter_cap, "Dumpi._SCATTER_WALKERS");
//    if (test == NULL) {
//        PyErr_SetString(PyExc_AttributeError, "Scatter pointer is NULL");
//        Py_XDECREF(scatter_cap);
//        Py_DECREF(module);
//        return NULL;
//    }

    static PyObject *gather_cap = PyCapsule_New((void *)Gather_Potentials, "Dumpi._GATHER_POTENTIALS", NULL);
    if (gather_cap == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Failed to create Gather pointer capsule");
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddObject(module, "_GATHER_POTENTIALS", gather_cap) < 0) {
        Py_XDECREF(gather_cap);
        Py_DECREF(module);
        return NULL;
    }

    static PyObject *gatherw_cap = PyCapsule_New((void *)Gather_Walkers, "Dumpi._GATHER_WALKERS", NULL);
    if (gatherw_cap == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Failed to create Gather walkers pointer capsule");
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddObject(module, "_GATHER_WALKERS", gatherw_cap) < 0) {
        Py_XDECREF(gatherw_cap);
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