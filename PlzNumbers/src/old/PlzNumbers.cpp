#include "PlzNumbers.hpp"

#include "Potators.hpp"
#include "PyAllUp.hpp"
#include <stdexcept>

int _LoadExtraArgs(
        ExtraBools &extra_bools, ExtraInts &extra_ints, ExtraFloats &extra_floats,
        PyObject* ext_bool, PyObject* ext_int, PyObject* ext_float
) {
    PyObject *iterator, *item;

    iterator = PyObject_GetIter(ext_bool);
    if (iterator == NULL) return 0;
    while ((item = PyIter_Next(iterator))) {
        extra_bools.push_back(_FromBool(item));
        Py_DECREF(item);
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return 0;

    iterator = PyObject_GetIter(ext_int);
    if (iterator == NULL) return 0;
    while ((item = PyIter_Next(iterator))) {
        extra_ints.push_back(_FromInt(item));
        Py_DECREF(item);
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return 0;


    iterator = PyObject_GetIter(ext_float);
    if (iterator == NULL) return 0;
    while ((item = PyIter_Next(iterator))) {
        extra_floats.push_back(_FromFloat(item));
        Py_DECREF(item);
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return 0;

    return 1;

}


PyObject *PlzNumbers_callPot(PyObject* self, PyObject* args ) {

    PyObject* atoms, *coords;
    PyObject* pot_function;
    PyObject* ext_bool, *ext_int, *ext_float;
    PyObject* bad_walkers_str;
    double err_val;
    int raw_array_pot, debug_print, retries;

    if (
         !PyArg_ParseTuple(args,
                 "OOOOdpppOOO",
                 &coords, &atoms, &pot_function, &bad_walkers_str,
                 &err_val, &raw_array_pot, &debug_print, &retries,
                 &ext_bool, &ext_int, &ext_float
           )
        ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;

    ExtraBools extra_bools; ExtraInts extra_ints; ExtraFloats extra_floats;
    if (!_LoadExtraArgs(
        extra_bools, extra_ints, extra_floats,
        ext_bool, ext_int, ext_float
        )) { return NULL; }

    PyObject* str = NULL;
    std::string bad_walkers_file =  _GetPyString(bad_walkers_str, str);
    Py_XDECREF(str);

//    if (debug_print) {
//        const char debug_str[500];
//        sprintf(debug_str, "Raw? %s\n", raw_array_pot ? "yes":"no")
//        _pyPrintStr(debug_str);
//    }

    Real_t pot;
    if (raw_array_pot) {
        FlatPotentialFunction pot_f = (FlatPotentialFunction) PyCapsule_GetPointer(pot_function, "_potential");
        FlatCoordinates walker_coords = _getWalkerFlatCoords(raw_data, 0, num_atoms);

        pot = _doopAPot(
                walker_coords,
                mattsAtoms,
                pot_f,
                bad_walkers_file,
                err_val,
                debug_print,
                extra_bools,
                extra_ints,
                extra_floats,
                retries
        );
    } else {
        PotentialFunction pot_f = (PotentialFunction) PyCapsule_GetPointer(pot_function, "_potential");
        Coordinates walker_coords = _getWalkerCoords(raw_data, 0, num_atoms);
        pot = _doopAPot(
                walker_coords,
                mattsAtoms,
                pot_f,
                bad_walkers_file,
                err_val,
                debug_print,
                extra_bools,
                extra_ints,
                extra_floats,
                retries
        );
    }

    PyObject *potVal = Py_BuildValue("f", pot);

    if (potVal == NULL) return NULL;

    return potVal;

}

PyObject *PlzNumbers_callPotVec( PyObject* self, PyObject* args ) {
    // vector version of callPot

    PyObject* coords, *atoms, *pot_function, *extra_args, *bad_walkers_file;
    double err_val;
    int raw_array_pot, vectorized_potential, debug_print;
    PyObject* manager;
    int use_openMP, use_TBB, retries;

    if ( !PyArg_ParseTuple(
            args,
            "OOOOOdppppOpp",
            &coords,
            &atoms,
            &pot_function,
            &extra_args,
            &bad_walkers_file,
            &err_val,
            &raw_array_pot,
            &vectorized_potential,
            &debug_print,
            &retries,
            &manager,
            &use_openMP,
            &use_TBB
            )
    ) return NULL;

    // Assumes we get n atom type names

//    if (debug_print) {
//        printf("this is super annoying...\n");
//    }

    Py_ssize_t num_atoms = PyObject_Length(atoms);
//    printf("how many of these are there...%d\n", num_atoms);
    if (PyErr_Occurred()) return NULL;
    Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

//    printf("like I figured this out before...\n");

    // we'll assume we have number of walkers X ncalls X number of atoms X 3
    PyObject *shape = PyObject_GetAttrString(coords, "shape");
    if (shape == NULL) return NULL;
    PyObject *ncalls_obj = PyTuple_GetItem(shape, 1);
    if (ncalls_obj == NULL) return NULL;
    Py_ssize_t ncalls = _FromInt(ncalls_obj);
    if (PyErr_Occurred()) return NULL;
    PyObject *num_walkers_obj = PyTuple_GetItem(shape, 0);
    if (num_walkers_obj == NULL) return NULL;
    Py_ssize_t num_walkers = _FromInt(num_walkers_obj);
    if (PyErr_Occurred()) return NULL;
//    Py_XDECREF(shape);

    // this thing should have the walker number as the slowest moving index then the number of the timestep
    // that way we'll really have the correct memory entering into our calls
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;

     // we load in the extra arguments that the potential can pass -- this bit of flexibility makes every
     // call a tiny bit slower, but allows us to not have to change this code constantly and recompile
    PyObject* ext_bool = PyTuple_GetItem(extra_args, 0);
    PyObject* ext_int = PyTuple_GetItem(extra_args, 1);
    PyObject* ext_float = PyTuple_GetItem(extra_args, 2);
    ExtraBools extra_bools; ExtraInts extra_ints; ExtraFloats extra_floats;
    if (!_LoadExtraArgs(
        extra_bools, extra_ints, extra_floats,
        ext_bool, ext_int, ext_float
        )) { return NULL; }

    // We can tell if MPI is active or not by whether COMM is None or not
    PotentialArray pot_vals;
    FlatPotentialFunction flat_pot;
    PotentialFunction pot;
    if (raw_array_pot) {
        pot = NULL;
        flat_pot = (FlatPotentialFunction) PyCapsule_GetPointer(pot_function, "_potential");
    } else {
        flat_pot = NULL;
        pot = (PotentialFunction) PyCapsule_GetPointer(pot_function, "_potential");
    }

//    printf("coords (%p) has %d refs...?\n", coords, Py_REFCNT(coords));

    bool main_core = true;
    if ( manager != Py_None ){
        PyObject *rank = PyObject_GetAttrString(manager, "world_rank");
        if (rank == NULL) { return NULL; }
        main_core = (_FromInt(rank) == 0);
        Py_XDECREF(rank);
    }
    PyObject* new_array;
    if (manager==Py_None) {
// //       printf("-_- y\n");
        pot_vals = _noMPIGetPot(
                pot,
                flat_pot,
                raw_data,
                mattsAtoms,
                ncalls,
                num_walkers,
                num_atoms,
                bad_walkers_file,
                err_val,
                debug_print,
                retries,
                extra_bools,
                extra_ints,
                extra_floats,
                use_openMP,
                use_TBB
        );
        new_array = _fillNumPyArray(pot_vals, ncalls, num_walkers);
//       new_array = _getNumPyArray(ncalls, num_walkers, "float");
    } else {
        pot_vals = _mpiGetPot(
                manager,
                pot,
                flat_pot,
                raw_data,
                mattsAtoms,
                ncalls,
                num_walkers,
                num_atoms,
                bad_walkers_file,
                err_val,
                debug_print,
                retries,
                extra_bools,
                extra_ints,
                extra_floats,
                use_openMP,
                use_TBB
        );
        if ( main_core) {
            new_array = _fillNumPyArray(pot_vals, num_walkers, ncalls);
        }
    }

//    printf("After all that, coords (%p) has %d refs and new_array has (%d)\n", coords, Py_REFCNT(coords), Py_REFCNT(new_array));

    if ( main_core ){
//        printf("._. %f %f %f (%d, %d)&(%d, %d)?\n",
//                pot_vals[0][0], pot_vals[1][2], pot_vals[2][4],
//                pot_vals.size(), pot_vals[0].size(),
//                num_walkers, ncalls
//                );
        return new_array;
    } else {
        Py_RETURN_NONE;
    }

}

PyObject *PlzNumbers_callPyPotVec( PyObject* self, PyObject* args ) {
    // vector version of callPot

    PyObject* atoms;
    PyObject* coords;
    PyObject* pot_function;
    PyObject* ext_args;
    PyObject* manager;
    if ( !PyArg_ParseTuple(args, "OOOOO",
                           &coords,
                           &atoms,
                           &pot_function,
                           &manager,
                           &ext_args
    ) ) return NULL;

    // MOST OF THIS BLOCK IS DIRECTLY COPIED FROM callPotVec

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    if (PyErr_Occurred()) return NULL;
    // But since we have a python potential we don't even pull them out...

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

//    printf("num calls %d num walkers %d num atoms %d\n", ncalls, num_walkers, num_atoms);
    // We can tell if MPI is active or not by whether COMM is None or not
    PyObject *pot_vals;
    if (manager == Py_None) {
        Py_RETURN_NONE;
    } else {
        pot_vals = _mpiGetPyPot(
                manager,
                pot_function,
                raw_data,
                atoms,
                ext_args,
                ncalls,
                num_walkers,
                num_atoms
        );
    }

    return pot_vals;

//    bool main_core = true;
//    if ( manager != Py_None ){
//        PyObject *rank = PyObject_GetAttrString(manager, "world_rank");
//        if (rank == NULL) { return NULL; }
//        main_core = (_FromInt(rank) == 0);
//        Py_XDECREF(rank);
//    }
//    if ( main_core ){
//        return pot_vals;
//    } else {
//        Py_RETURN_NONE;
//    }

}

// PYTHON WRAPPER EXPORT

static PyMethodDef PlzNumbersMethods[] = {
    {"rynaLovesPoots", PlzNumbers_callPot, METH_VARARGS, "calls a potential on a single walker"},
    {"rynaLovesPootsLots", PlzNumbers_callPotVec, METH_VARARGS, "calls a potential on a vector of walkers"},
    {"rynaLovesPyPootsLots", PlzNumbers_callPyPotVec, METH_VARARGS, "calls a _python_ potential on a vector of walkers"},
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