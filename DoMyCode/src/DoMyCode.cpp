#include "DoMyCode.hpp"

#include "PyAllUp.hpp"
#include <stdexcept>

namespace DoMyCode {

    static int DEBUG_LEVEL = 0;
    static int DEBUG_ALL = 10;
    void _GetDebugLevel(PyObject* mod) {
        PyObject *debugLevelObj = PyObject_GetAttrString(mod, "DEBUG_LEVEL");
        DEBUG_LEVEL = _FromInt(debugLevelObj);
    }
    int DebugPrint(int level, const char *fmt, ...) {
        if (level <= DEBUG_LEVEL) {
            va_list args;
            va_start(args, fmt);
            int fmt_len = strlen(fmt);
            char* new_fmt = (char*) malloc(sizeof(char)*(fmt_len+10));
            sprintf(new_fmt, "%s\n", fmt);
            int res = vprintf(new_fmt, args);
            free((void*)new_fmt);
//            fflush(stdout);
            return res;
        } else {
            return 0;
        }
    }
    const char* _GetPyRepr(PyObject* obj, PyObject *str) {
//        DebugPrint(DEBUG_ALL, "Extracting repr from %p", obj);
        const char* out;
        if (obj == NULL) {
            out = "<NULL>";
        } else {
            PyObject *repr = PyObject_Repr(obj);
            out = _GetPyString(repr, str);
        }
//        DebugPrint(DEBUG_ALL, "Got %s from %p", out, obj);
        return out;
    }

    PyObject *DoMyCode_distributeWalkers(PyObject *self, PyObject *args) {

        _GetDebugLevel(self); // Sets the debug level

        PyObject *coords, *manager;

        if (!PyArg_ParseTuple(args, "OO", &coords, &manager)) { return NULL; }

        // Figure out how many things to send/get
        PyObject *ws = PyObject_GetAttrString(manager, "world_size");
        int world_size = _FromInt(ws);
        Py_XDECREF(ws);
        PyObject *wr = PyObject_GetAttrString(manager, "world_rank");
        int world_rank = _FromInt(wr);
        Py_XDECREF(wr);

        DebugPrint(DEBUG_ALL, "(Sending on %d): To start, coords (%p) has %d refs and manager has %d",
                world_rank,
                coords,
                Py_REFCNT(coords),
                Py_REFCNT(manager)
        );

        PyObject *shape = PyObject_GetAttrString(coords, "shape");
        if (shape == NULL) return NULL;

        DebugPrint(DEBUG_ALL, "(On %d): Extracted shape", world_rank);

        PyObject *num_walkers_obj = PyTuple_GetItem(shape, 0);
        if (num_walkers_obj == NULL) return NULL;
        Py_ssize_t num_walkers = _FromInt(num_walkers_obj);
        if (PyErr_Occurred()) return NULL;

        DebugPrint(DEBUG_ALL, "(On %d): Extracted num walkers, %d", world_rank, num_walkers);

        int num_walkers_per_core;
        if (world_rank == 0) {
            num_walkers_per_core = num_walkers / world_size;
        } else {
            num_walkers_per_core = num_walkers;
        }

        DebugPrint(DEBUG_ALL, "(On %d): Computed walkers/core, %d", world_rank, num_walkers_per_core);

        if (DEBUG_LEVEL >= DEBUG_ALL) {
            PyObject *tmp = NULL;
            const char *coords_shape = _GetPyRepr(shape, tmp);
            DebugPrint(
                    DEBUG_ALL,
                    "(On %d) Sending coords with shape %s over world_size %d",
                    world_rank, coords_shape, world_size
            );
            Py_XDECREF(tmp);
        }

        PyObject *natoms_obj = PyTuple_GetItem(shape, 1);
        if (natoms_obj == NULL) return NULL;
        Py_ssize_t num_atoms = _FromInt(natoms_obj);
        if (PyErr_Occurred()) return NULL;
        Py_XDECREF(shape);

        DebugPrint(DEBUG_ALL, "(On %d): Extracted num atoms", world_rank);

        // Assumes number of walkers X number of atoms X 3
        RawWalkerBuffer raw_data;
        if (world_rank == 0) {
            raw_data = _GetDoubleDataArray(coords);
            if (raw_data == NULL) return NULL;
        } else {
            raw_data = NULL;
        }

        if (DEBUG_LEVEL >= DEBUG_ALL) {
            PyObject *array_size = PyObject_GetAttrString(coords, "nbytes");
            Py_ssize_t byte_size = _FromInt(array_size);
            DebugPrint(DEBUG_ALL,
                       "(On %d): Got raw data array of size %lu",
                       world_rank,
                       byte_size
//                sizeof(raw_data)*num_walkers_per_core*world_size*num_atoms*3
            );
            Py_XDECREF(array_size);
        }

        PyObject *little_walkers;
        little_walkers = _getNumPyArray(num_walkers_per_core, num_atoms, 3, "float");
//        if (world_rank == 0) {
//            little_walkers = _getNumPyArray(num_walkers_per_core, num_atoms, 3, "float");
//        } else {
//            little_walkers =  coords; // reuse the memory because why not...? -- turns out you get subtle memory corruption :yay:
//        }
        if (little_walkers == NULL) return NULL;
        DebugPrint(DEBUG_ALL, "(On %d): Extracting little_walkers buffer", world_rank);
        RawWalkerBuffer walker_buf = _GetDoubleDataArray(little_walkers);
        if (world_rank > 0) { raw_data = walker_buf; } // just to shut OMPI up...

        if (DEBUG_LEVEL >= DEBUG_ALL) {
            PyObject *array_size = PyObject_GetAttrString(little_walkers, "nbytes");
            Py_ssize_t byte_size = _FromInt(array_size);
            DebugPrint(DEBUG_ALL,
                       "(On %d): Chunking into array of size %lu",
                       world_rank,
                       byte_size
//                sizeof(raw_data)*num_walkers_per_core*world_size*num_atoms*3
            );
            Py_XDECREF(array_size);
        }

        if (DEBUG_LEVEL >= DEBUG_ALL) {
            PyObject *tmp = NULL;
            PyObject *lil_shape = PyObject_GetAttrString(little_walkers, "shape");
            if (lil_shape == NULL) return NULL;
            const char *lil_shape_str = _GetPyRepr(lil_shape, tmp);
            DebugPrint(DEBUG_ALL, "(On %d): Built lil_walkers object with shape %s", world_rank, lil_shape_str);
            Py_XDECREF(tmp);
            Py_XDECREF(lil_shape);
        }

        PyObject *scatter = PyObject_GetAttrString(manager, "scatter");
        ScatterFunction scatter_walkers = (ScatterFunction) PyCapsule_GetPointer(scatter, "Dumpi._SCATTER_WALKERS");
//        Py_XDECREF(PyObject_CallMethod(manager, "wait", NULL));
        DebugPrint(DEBUG_ALL, "(On %d): Scattering walkers from PID %d", world_rank, getpid());
//        Py_XDECREF(PyObject_CallMethod(manager, "wait", NULL));
        int scattered = scatter_walkers(
                manager,
                raw_data,  // raw data buffer to chunk up
                num_walkers_per_core,
                num_atoms * 3, // three coordinates per atom per num_atoms per walker
                walker_buf // raw array to write into
        );
        Py_XDECREF(scatter);
        if (!scattered) { return NULL; }

//        if (world_rank == 0) {
//            // I guess we need to add a ref...?
//            Py_INCREF(little_walkers);
//        }

        DebugPrint(DEBUG_ALL, "(On %d): Scattered walkers, coords has %d refs, little_walkers has %d",
                   world_rank,
                   Py_REFCNT(coords),
                   Py_REFCNT(little_walkers)
        );

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
            big_poots = _getNumPyArray(num_steps * world_size, num_walkers_per_core, "float");
            if (big_poots == NULL) { return NULL; }
            poot_buf = _GetDoubleDataArray(big_poots);
        } else {
            poot_buf = pot_data;
        }

        DebugPrint(DEBUG_ALL, "(On %d): Filling in potential buffer", world_rank);

        PyObject *gather_pots = PyObject_GetAttrString(manager, "gather");
        GatherFunction gather = (GatherFunction) PyCapsule_GetPointer(gather_pots, "Dumpi._GATHER_POTENTIALS");
        if (gather == NULL) {
            PyErr_SetString(PyExc_AttributeError, "Couldn't get gather pointer");
            return NULL;
        }
        int success = gather(
                manager,
                pot_data,
                num_steps * num_walkers_per_core, // number of walkers fed in
                poot_buf // buffer to get the potential values back
        );
        Py_XDECREF(gather_pots);

        if (!success) { return NULL; }

        if (world_rank == 0) {
            if (DEBUG_LEVEL < DEBUG_ALL) {
                PyObject* shape = PyObject_GetAttrString(big_poots, "shape");
                if (shape == NULL) return NULL;
                PyObject* tmp = NULL;
                const char* shape_str = _GetPyRepr(shape, tmp);
                DebugPrint(DEBUG_ALL, "(On %d): Filled in potential buffer of shape %s", world_rank, shape_str);
                Py_XDECREF(shape); Py_XDECREF(tmp);
            }
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
            if (big_walkers == NULL) { return NULL; }
            walker_buf = _GetDoubleDataArray(big_walkers);
        }

        DebugPrint(DEBUG_ALL, "(On %d): Filling in walker buffer", world_rank);

        PyObject *gather_walkers = PyObject_GetAttrString(manager, "gather_walkers");
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
            if (DEBUG_LEVEL < DEBUG_ALL) {
                PyObject* shape = PyObject_GetAttrString(big_walkers, "shape");
                if (shape == NULL) return NULL;
                PyObject* tmp = NULL;
                const char* shape_str = _GetPyRepr(shape, tmp);
                DebugPrint(DEBUG_ALL, "(On %d): Filled in walker buffer of shape %s", world_rank, shape_str);
                Py_XDECREF(shape); Py_XDECREF(tmp);
            }
            return big_walkers;
        } else {
            Py_RETURN_NONE;
        }
    }

    PyObject *DoMyCode_getWalkersAndPots(PyObject *self, PyObject *args) {

        _GetDebugLevel(self);

        PyObject *coords, *potentials, *weights, *manager;

        if (!PyArg_ParseTuple(args, "OOOO", &coords, &potentials, &weights, &manager)) return NULL;

        // Figure out how many things to send/get
        PyObject *ws = PyObject_GetAttrString(manager, "world_size");
        int world_size = _FromInt(ws);
        Py_XDECREF(ws);
        PyObject *wr = PyObject_GetAttrString(manager, "world_rank");
        int world_rank = _FromInt(wr);
        Py_XDECREF(wr);

        DebugPrint(DEBUG_ALL,
                   "(Gathering on %d): To start, coords has %d refs, manager has %d, potentials has %d, and weights has %d",
                   world_rank,
                   Py_REFCNT(coords),
                   Py_REFCNT(manager),
                   Py_REFCNT(potentials),
                   Py_REFCNT(weights)
        );

        PyObject *shape = PyObject_GetAttrString(coords, "shape");
        if (shape == NULL) return NULL;

        if (DEBUG_LEVEL >= DEBUG_ALL) {
            PyObject *tmp = NULL;
            const char *coords_shape = _GetPyRepr(shape, tmp);
            DebugPrint(
                    DEBUG_ALL,
                    "(On %d): getting coords with shape %s",
                    world_rank, coords_shape, world_size
            );
            Py_XDECREF(tmp);
        }

        PyObject *num_wpc_obj = PyTuple_GetItem(shape, 0);
        if (num_wpc_obj == NULL) return NULL;
        Py_ssize_t num_walker_per_core = _FromInt(num_wpc_obj);
        if (PyErr_Occurred()) return NULL;

        DebugPrint(DEBUG_ALL, "(Gathering on %d): Got walkers/core %lu", world_rank, num_walker_per_core);

        PyObject *natoms_obj = PyTuple_GetItem(shape, 1);
        if (natoms_obj == NULL) return NULL;
        Py_ssize_t num_atoms = _FromInt(natoms_obj);
        if (PyErr_Occurred()) return NULL;
        Py_XDECREF(shape);

        DebugPrint(DEBUG_ALL, "(Gathering on %d): Got num atoms %lu", world_rank, num_atoms);

        PyObject *pot_shape = PyObject_GetAttrString(potentials, "shape");
        if (shape == NULL) return NULL;

        PyObject *num_steps_obj = PyTuple_GetItem(pot_shape, 0);
        if (num_steps_obj == NULL) return NULL;
        Py_ssize_t num_steps = _FromInt(num_steps_obj);
        if (PyErr_Occurred()) return NULL;
        Py_XDECREF(pot_shape);

        // Assumes number of walkers X number of atoms X 3
        double *coord_data = _GetDoubleDataArray(coords);
        if (coord_data == NULL) return NULL;
        PyObject *big_walkers = _gatherWalkers(
                manager,
                world_rank, world_size,
                coord_data,
                num_walker_per_core,
                num_atoms
        );
        if (big_walkers == NULL) return NULL;

        double *pot_data = _GetDoubleDataArray(potentials);
        if (pot_data == NULL) return NULL;
        PyObject *big_poots = _gatherPotentials(
                manager,
                world_rank, world_size,
                pot_data,
                num_steps,
                num_walker_per_core
        );
        if (big_poots == NULL) return NULL;

        DebugPrint(DEBUG_ALL, "(On %d): Sent potential values", world_rank);

        PyObject *big_weights;
        if (weights != Py_None) {
            double *weights_data = _GetDoubleDataArray(weights);
            if (weights_data == NULL) return NULL;
            big_weights = _gatherPotentials(
                    manager,
                    world_rank, world_size,
                    weights_data,
                    1,
                    num_walker_per_core
            );
            if (big_weights == NULL) return NULL;
        } else { ;
            big_weights = Py_None;
        }

        if (world_rank == 0) {
            PyObject *ret;
            if (weights == Py_None) {
                ret = Py_BuildValue("(OO)", big_walkers, big_poots);
                Py_XDECREF(big_walkers);
                Py_XDECREF(big_poots);
            } else {
                ret = Py_BuildValue("(OOO)", big_walkers, big_poots, big_weights);
                Py_XDECREF(big_walkers);
                Py_XDECREF(big_poots);
                Py_XDECREF(big_weights);
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
            {"sendFriends",        DoMyCode_distributeWalkers, METH_VARARGS, "distributes walkers to the nodes"},
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

    PyMODINIT_FUNC PyInit_DoMyCode(void) {
        PyObject* mod = PyModule_Create(&DoMyCodeModule);
        PyObject* debug_level = Py_BuildValue("i", DEBUG_LEVEL);
        if (PyModule_AddObject(mod, "DEBUG_LEVEL", debug_level) < 0) {
            Py_XDECREF(mod);
            Py_DECREF(debug_level);
            return NULL;
        }

        return mod;

    }
#else

    PyMODINIT_FUNC initDoMyCode(void)
    {
        (void) Py_InitModule("DoMyCode", DoMyCodeMethods);
    }

#endif

}