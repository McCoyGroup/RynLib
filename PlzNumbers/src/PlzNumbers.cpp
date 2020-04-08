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

    PyObject* atoms;
    PyObject* coords;
    PyObject* pot_function;
    PyObject* ext_bool, *ext_int, *ext_float;
    const char* bad_walkers_str;
    double err_val;
    bool raw_array_pot;

    if (
         !PyArg_ParseTuple(args, "OOOOdpOOO",
           &atoms, &coords, &pot_function, &bad_walkers_str, &err_val, &raw_array_pot,
           &ext_bool, &ext_int, &ext_float
           )
        ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

//    PyObject *coordString = PyObject_Repr(coords);
//    PyObject *str;
//    printf("%s", _GetPyString(coordString, str));
//    Py_XDECREF(coordString);
//    Py_XDECREF(str);

    // Assumes number of walkers X number of atoms X 3
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;
    Coordinates walker_coords = _getWalkerCoords(raw_data, 0, num_atoms);

    ExtraBools extra_bools; ExtraInts extra_ints; ExtraFloats extra_floats;
    if (!_LoadExtraArgs(
        extra_bools, extra_ints, extra_floats,
        ext_bool, ext_int, ext_float
        )) { return NULL; }

    PotentialFunction pot_f = (PotentialFunction) PyCapsule_GetPointer(pot_function, "_potential");

    std::string bad_walkers_file = bad_walkers_str;

    Real_t pot = _doopAPot(
            walker_coords,
            mattsAtoms,
            pot_f,
            bad_walkers_file,
            err_val,
            extra_bools,
            extra_ints,
            extra_floats
    );

    PyObject *potVal = Py_BuildValue("f", pot);

    if (potVal == NULL) return NULL;

    return potVal;

}

PyObject *PlzNumbers_callPotVec( PyObject* self, PyObject* args ) {
    // vector version of callPot

    PyObject* atoms;
    PyObject* coords;
    PyObject* pot_function;
    PyObject* bad_walkers_file;
    PyObject* ext_bool, *ext_int, *ext_float;
    double err_val;
    bool raw_array_pot, vectorized_potential;
    PyObject* manager;
    if ( !PyArg_ParseTuple(args, "OOOOdppOOOO",
            &atoms,
            &coords,
            &pot_function,
            &bad_walkers_file,
            &err_val,
            &raw_array_pot,
            &vectorized_potential,
            &manager,
            &ext_bool,
            &ext_int,
            &ext_float
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


     // we load in the extra arguments that the potential can pass -- this bit of flexibility makes every
     // call a tiny bit slower, but allows us to not have to change this code constantly and recompile

    ExtraBools extra_bools; ExtraInts extra_ints; ExtraFloats extra_floats;
    if (!_LoadExtraArgs(
        extra_bools, extra_ints, extra_floats,
        ext_bool, ext_int, ext_float
        )) { return NULL; }

    // We can tell if MPI is active or not by whether COMM is None or not
    PotentialFunction pot = (PotentialFunction) PyCapsule_GetPointer(pot_function, "_potential");
    PotentialArray pot_vals;
    if (manager == Py_None) {
        pot_vals = _noMPIGetPot(
                pot,
                raw_data,
                mattsAtoms,
                ncalls,
                num_walkers,
                num_atoms,
                bad_walkers_file,
                err_val,
                vectorized_potential,
                extra_bools,
                extra_ints,
                extra_floats
                );
    } else {
        pot_vals = _mpiGetPot(
                manager,
                pot,
                raw_data,
                mattsAtoms,
                ncalls,
                num_walkers,
                num_atoms,
                bad_walkers_file,
                err_val,
                vectorized_potential,
                extra_bools,
                extra_ints,
                extra_floats
        );
    }

    bool main_core = true;
    if ( manager != Py_None ){
        PyObject *rank = PyObject_GetAttrString(manager, "world_rank");
        if (rank == NULL) { return NULL; }
        main_core = (_FromInt(rank) == 0);
        Py_XDECREF(rank);
    }
    if ( main_core ){
        return _fillNumPyArray(pot_vals, num_walkers, ncalls);
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
                           &atoms,
                           &coords,
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