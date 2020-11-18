/*
 * This is a template file to make it easy to use a Fortran potential
 * To start, replace <LibName> with the name of the library to be compiled (i.e. the name of the folder this is in)
 * Then, replace <LibFunction> with the name of the function to call
 * Finally, replace <CallSignature> with the call signature that Fortran expects, keeping in mind that Fortran
 *   wants only _references_ to memory (i.e. pointers)
 * 
 */
#include "<LibName>.hpp" // We also need to edit <LibName>.hpp
#include "RynTypes.hpp"

Real_t <LibName>_<LibFunction>(
    FlatCoordinates coords,
    Names atoms,
    ExtraBools extra_bools,
    ExtraInts extra_ints,
    ExtraFloats extra_floats
    ) {

    // Load extra args (if necessary)
    // We allow for a list of extra bools, ints, and floats.
    // We could add to the type-list, probably, but I don't want to be bothered with it at the moment and
    // after this starts to get used more and more it'll probably be too late to change anything, too...
    // So that's probably what we're stuck with.
    // Here's an example: on the python side we'd call pot(coords, atoms, 55.2) and then on this side we'd have
    // int example_parameter = extra_ints[0]; // example_parameter = 55.2

    // This puts the coordinates in an array that can be fed to Fortran
    RawWalkerBuffer raw_coords = coords.data();

    // We need to initialize a double in memory so that Fortran can fill that memory
    Real_t energy = -1000; // I set it to something crazy so we can easily know if this worked out or not

    // Do the actual Fortran call
    // e.g. getpot(raw_coords, &energy) in the absolute simplest case
    <LibFunction>(<CallSignature>);

    return energy;

}

// From here on down is python boiler-plate
static PyObject* <LibName>_<LibFunction>Wrapper = PyCapsule_New((void *)<LibName>_<LibFunction>, "_potential", NULL);

bool _AttachCapsuleToModule(PyObject* module, PyObject* capsule, const char* name) {
    bool i_did_good = (PyModule_AddObject(module, name, capsule) == 0);
    if (!i_did_good) {
        Py_XDECREF(capsule);
        Py_DECREF(module);
    }

    return i_did_good;
}

static PyMethodDef <LibName>Methods[] = {
        {NULL, NULL, 0, NULL}
};

// TODO: ADD IN SOMETHING THAT LETS US GET THE ARGUMENT NAMES DIRECTLY FROM THE LIB

#if PY_MAJOR_VERSION > 2

const char <LibName>_doc[] = "<LibName> uses the Bowman CH5+ surface to return energies for structures";
static struct PyModuleDef <LibName>Module = {
    PyModuleDef_HEAD_INIT,
    "<LibName>",   /* name of module */
    <LibName>_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    <LibName>Methods
};

PyMODINIT_FUNC PyInit_<LibName>(void)
{
    PyObject *m;
    m = PyModule_Create(&<LibName>Module);
    if (m == NULL) {
        return NULL;
    }

    if (!_AttachCapsuleToModule(m, <LibName>_<LibFunction>Wrapper, "_potential")) { return NULL; }

    return m;
}

#else

PyMODINIT_FUNC <LibName>(void)
{
    PyObject *m;
    m = Py_InitModule("<LibName>", <LibName>Methods);
    if (m == NULL) {
    return NULL
    }

    _AttachCapsuleToModule(m, <LibName>_<LibFunction>Wrapper);
}

#endif