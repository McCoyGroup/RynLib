#include "DoMyCode.hpp"

#include "PyAllUp.hpp"
#include <stdexcept>

PyObject *DoMyCode_distributeWalkers(PyObject* self, PyObject* args ) {

    PyObject *coords, *manager;

    if (
         !PyArg_ParseTuple(args,
                 "OO",
                 &coords, &manager
           )
        ) return NULL;

//    printf("To start, coords (%p) has %d refs and manager has %d\n", coords, Py_REFCNT(coords), Py_REFCNT(manager));

    // Figure out how many things to send/get
    PyObject* ws = PyObject_GetAttrString(manager, "world_size");
    int world_size = _FromInt(ws);
    Py_XDECREF(ws);
    PyObject* wr = PyObject_GetAttrString(manager, "world_rank");
    int world_rank = _FromInt(wr);
    Py_XDECREF(wr);

    PyObject *shape = PyObject_GetAttrString(coords, "shape");
    if (shape == NULL) return NULL;

    PyObject *num_walkers_obj = PyTuple_GetItem(shape, 0);
    if (num_walkers_obj == NULL) return NULL;
    Py_ssize_t num_walkers = _FromInt(num_walkers_obj);
    if (PyErr_Occurred()) return NULL;

    int num_walkers_per_core;
    if (world_rank == 0) {
        num_walkers_per_core = num_walkers / world_size;
    } else {
        num_walkers_per_core = num_walkers;
    }

    PyObject *natoms_obj = PyTuple_GetItem(shape, 1);
    if (natoms_obj == NULL) return NULL;
    Py_ssize_t num_atoms = _FromInt(natoms_obj);
    if (PyErr_Occurred()) return NULL;
    Py_XDECREF(shape);

    // Assumes number of walkers X number of atoms X 3
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;

    PyObject *little_walkers;
    if (world_rank == 0) {
        little_walkers = _getNumPyArray(num_walkers_per_core, num_atoms, 3, "float");
    } else {
        little_walkers = coords; // reuse the memory because why not...?
    }
    if (little_walkers == NULL) return NULL;
    double *walker_buf = _GetDoubleDataArray(little_walkers);

    PyObject* scatter = PyObject_GetAttrString(manager, "scatter");
    ScatterFunction scatter_walkers = (ScatterFunction) PyCapsule_GetPointer(scatter, "Dumpi._SCATTER_WALKERS");
    scatter_walkers(
            manager,
            raw_data,  // raw data buffer to chunk up
            num_walkers_per_core,
            num_atoms * 3, // three coordinates per atom per num_atoms per walker
            walker_buf // raw array to write into
    );
    Py_XDECREF(scatter);

//    printf("  after the scatter, coords (%p) has %d refs...?\n", coords, Py_REFCNT(coords));

    return little_walkers;

}

PyObject *_gatherPotentials(
        PyObject *manager,
        int world_rank, int world_size,
        RawPotentialBuffer pot_data,
        int num_steps,
        int num_walkers_per_core
        ) {
    PyObject *big_poots = NULL;
    double *poot_buf = NULL;
    if (world_rank == 0) {
        big_poots = _getNumPyArray(num_steps*world_size, num_walkers_per_core, "float");
        poot_buf = _GetDoubleDataArray(big_poots);
    }

    PyObject* gather_pots = PyObject_GetAttrString(manager, "gather");
    GatherFunction gather = (GatherFunction) PyCapsule_GetPointer(gather_pots, "Dumpi._GATHER_POTENTIALS");
    if (gather == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Couldn't get gather pointer");
        return NULL;
    }
    gather(
            manager,
            pot_data,
            num_steps * num_walkers_per_core, // number of walkers fed in
            poot_buf // buffer to get the potential values back
    );
    Py_XDECREF(gather_pots);

    if (world_rank == 0) {
        return big_poots;
    } else {
        Py_RETURN_NONE;
    }

}
PyObject *_gatherWalkers(
        PyObject *manager,
        int world_rank, int world_size,
        RawWalkerBuffer coord_data,
//        int num_steps,
        int num_walkers_per_core,
        int num_atoms
        ) {

    PyObject *big_walkers = NULL;
    double *walker_buf = NULL;
    if (world_rank == 0) {
        big_walkers = _getNumPyArray(num_walkers_per_core * world_size, num_atoms, 3, "float");
        walker_buf = _GetDoubleDataArray(big_walkers);
    }

    PyObject* gather_walkers = PyObject_GetAttrString(manager, "gather_walkers");
    GatherWalkerFunction gather_w = (GatherWalkerFunction) PyCapsule_GetPointer(gather_walkers, "Dumpi._GATHER_WALKERS");
    if (gather_w == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Couldn't get gather pointer");
        return NULL;
    }
    int success = gather_w(
            manager,
            coord_data,  // raw data buffer to chunk up
            num_walkers_per_core,
            num_atoms * 3, // three coordinates per atom per num_atoms per walker
            walker_buf // raw array to write into
    );
    Py_XDECREF(gather_walkers);

    if (!success) { return NULL; }

    if (world_rank == 0) {
        return big_walkers;
    } else {
        Py_RETURN_NONE;
    }
}

PyObject *DoMyCode_getWalkersAndPots(PyObject* self, PyObject* args ) {

    PyObject *coords, *potentials, *weights, *manager;

    if (!PyArg_ParseTuple(args, "OOOO", &coords, &potentials, &weights, &manager)) return NULL;

    // Figure out how many things to send/get
    PyObject* ws = PyObject_GetAttrString(manager, "world_size");
    int world_size = _FromInt(ws);
    Py_XDECREF(ws);
    PyObject* wr = PyObject_GetAttrString(manager, "world_rank");
    int world_rank = _FromInt(wr);
    Py_XDECREF(wr);

//    printf("Before the gather, coords (%p) has %d refs and potentials (%p) has %d\n",
//            coords, Py_REFCNT(coords),
//            potentials, Py_REFCNT(potentials)
//            );

    PyObject *shape = PyObject_GetAttrString(coords, "shape");
    if (shape == NULL) return NULL;

    PyObject *num_wpc_obj = PyTuple_GetItem(shape, 0);
    if (num_wpc_obj == NULL) return NULL;
    Py_ssize_t num_walker_per_core = _FromInt(num_wpc_obj);
    if (PyErr_Occurred()) return NULL;

    PyObject *natoms_obj = PyTuple_GetItem(shape, 1);
    if (natoms_obj == NULL) return NULL;
    Py_ssize_t num_atoms = _FromInt(natoms_obj);
    if (PyErr_Occurred()) return NULL;
    Py_XDECREF(shape);

    PyObject *pot_shape = PyObject_GetAttrString(potentials, "shape");
    if (shape == NULL) return NULL;

    PyObject *num_steps_obj = PyTuple_GetItem(pot_shape, 0);
    if (num_steps_obj == NULL) return NULL;
    Py_ssize_t num_steps = _FromInt(num_steps_obj);
    if (PyErr_Occurred()) return NULL;
    Py_XDECREF(pot_shape);

    // Assumes number of walkers X number of atoms X 3
    double* coord_data = _GetDoubleDataArray(coords);
    if (coord_data == NULL) return NULL;
    PyObject* big_walkers = _gatherWalkers(
            manager,
            world_rank, world_size,
            coord_data,
            num_walker_per_core,
            num_atoms
            );
    if (big_walkers == NULL) return NULL;

//    printf("...sent walker (%d)\n", world_rank);

    double* pot_data = _GetDoubleDataArray(potentials);
    if (pot_data == NULL) return NULL;
    PyObject* big_poots = _gatherPotentials(
            manager,
            world_rank, world_size,
            pot_data,
            num_steps,
            num_walker_per_core
    );
    if (big_poots == NULL) return NULL;

//    printf("...sent poots (%d)\n", world_rank);

    PyObject* big_weights;
    if (weights != Py_None) {
        double* weights_data = _GetDoubleDataArray(weights);
        if (weights_data == NULL) return NULL;
        big_weights = _gatherPotentials(
            manager,
            world_rank, world_size,
            weights_data,
            1,
            num_walker_per_core
        );
        if (big_weights == NULL) return NULL;
    } else {;
        big_weights = Py_None;
    }
    
//    printf("To end, coords (%p) has %d refs and manager has %d\n", coords,
//            Py_REFCNT(coords), Py_REFCNT(manager));


    if (world_rank == 0) {
        PyObject *ret;
        if (weights == Py_None) {
            ret = Py_BuildValue("(OO)", big_walkers, big_poots);
            Py_XDECREF(big_walkers); Py_XDECREF(big_poots);
        } else {
            ret = Py_BuildValue("(OOO)", big_walkers, big_poots, big_weights);
            Py_XDECREF(big_walkers); Py_XDECREF(big_poots); Py_XDECREF(big_weights);
        };
        if (ret == NULL) { return NULL; }
        return ret;
    } else {
        Py_RETURN_NONE;
    }

}

//for dying in eliminated_walkers:  # gotta do it iteratively to get the max_weight_walker right...
//        cloning = np.argmax(weights)
//# print(cloning)
//parents[dying] = parents[cloning]
//walkers[dying] = walkers[cloning]
//weights[dying] = weights[cloning] / 2.0
//weights[cloning] /= 2.0

static PyMethodDef DoMyCodeMethods[] = {
    {"sendFriends", DoMyCode_distributeWalkers, METH_VARARGS, "distributes walkers to the nodes"},
    {"getFriendsAndPoots", DoMyCode_getWalkersAndPots, METH_VARARGS, "gets walkers and potentials from the nodes"},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION > 2

const char DoMyCode_doc[] = "DoMyCode manages the distribution of walkers and return of the potential values w/ MPI";
static struct PyModuleDef DoMyCodeModule = {
    PyModuleDef_HEAD_INIT,
    "DoMyCode",   /* name of module */
    DoMyCode_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    DoMyCodeMethods
};

PyMODINIT_FUNC PyInit_DoMyCode(void)
{
    return PyModule_Create(&DoMyCodeModule);
}
#else

PyMODINIT_FUNC initDoMyCode(void)
{
    (void) Py_InitModule("DoMyCode", DoMyCodeMethods);
}

#endif