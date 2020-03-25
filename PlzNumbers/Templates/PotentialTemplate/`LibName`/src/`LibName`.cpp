//
// Created by Mark Boyer on 1/31/20.
//

#include "`LibName`.hpp"
#include "RynTypes.hpp"

Real_t `LibName`_Potential(
    const Coordinates coords,
    const Names atoms,
    const ExtraBools extra_bools,
    const ExtraInts extra_ints,
    const ExtraFloats extra_floats
    ) {

    // Load extra args (if necessary)
    `PotentialLoadExtraBools`
    `PotentialLoadExtraInts`
    `PotentialLoadExtraFloats`

    bool raw_pot = `OldStylePotential`;
    if (raw_pot) {
        // Copy data into a single array
        int nels = coords.size() * coords[0].size();
        std::vector<Real_t> long_vec (nels);
        nels = 0;
        std::vector<Real_t> sub_vec;
        for (unsigned long v=0; v<coords.size(); v++) {
            sub_vec = coords[v];
            long_vec.insert(long_vec.end(), sub_vec.begin(), sub_vec.end());
        }
        RawWalkerBuffer raw_coords = long_vec.data();

        std::string long_atoms;
        for (unsigned long v=0; v<atoms.size(); v++) {
            long_atoms = long_atoms + atoms[v] + " ";
        }
        const char* raw_atoms = long_atoms.data();
    }

    return `PotentialCall`;
}

static PyObject* `LibName`_PotentialWrapper = PyCapsule_New((void *)`LibName`_Potential, "_potential", NULL);

bool _AttachCapsuleToModule(PyObject* module, PyObject* capsule) {
    bool i_did_good = (PyModule_AddObject(module, "_potential", capsule) == 0);
    if (!i_did_good) {
        printf("oops");
        Py_XDECREF(capsule);
        Py_DECREF(module);
    }

    return i_did_good;
}

static PyMethodDef `LibName`Methods[] = {
        {NULL, NULL, 0, NULL}
};

// TODO: ADD IN SOMETHING THAT LETS US GET THE ARGUMENT NAMES DIRECTLY FROM THE LIB

#if PY_MAJOR_VERSION > 2

const char `LibName`_doc[] = "`LibName` provides a potential";
static struct PyModuleDef `LibName`Module = {
    PyModuleDef_HEAD_INIT,
    "`LibName`",   /* name of module */
    `LibName`_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    `LibName`Methods
};

PyMODINIT_FUNC PyInit_`LibName`(void)
{
    PyObject *m;
    m = PyModule_Create(&`LibName`Module);
    if (m == NULL) {
        return NULL;
    }

    if (!_AttachCapsuleToModule(m, `LibName`_PotentialWrapper)) { return NULL; }

    return m;
}

#else

PyMODINIT_FUNC init`LibName`(void)
{
    PyObject *m;
    m = Py_InitModule("`LibName`", `LibName`Methods);
    if (m == NULL) {
    return NULL
    }

    _AttachCapsuleToModule(m, `LibName`_PotentialWrapper);
}

#endif