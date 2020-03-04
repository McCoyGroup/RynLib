//
// Created by Mark Boyer on 1/31/20.
//

#include "`LibName`.hpp"
#include "RynTypes.hpp"

Real_t `LibName`_Potential(const Coordinates coords, const Names atoms) {
    return `LibNameOfPotential`(coords, atoms);
}

static PyObject* `LibName`_PotentialWrapper = PyCapsule_New((void *)`LibName`_Potential, "`LibName`._potential", NULL);

bool _AttachCapsuleToModule(PyObject* module, PyObject* capsule) {
    bool i_did_good = (PyModule_AddObject(module, "_potential", capsule) < 0);
    if (!i_did_good) {
        Py_XDECREF(capsule);
        Py_DECREF(module);
    }

    return i_did_good;
}

static PyMethodDef `LibName`Methods[] = {
//        {"pootTheToot", `LibName`_Potential, METH_VARARGS, "calls the potential on a single walker"},
        {NULL, NULL, 0, NULL}
};


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

    if (!_AttachCapsuleToModule(m, `LibName`_PotentialWrapper)) {return NULL;}

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