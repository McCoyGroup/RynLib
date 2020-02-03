//
// Created by Mark Boyer on 1/31/20.
//

#include "`LibName`Pot.hpp"
#include "RynTypes.hpp"

Real_t `LibName`_pot(const Coordinates coords, const Names atoms) {
    return `LibNameOfPotential`(coords, atoms);
}

static PyMethodDef `LibName`Methods[] = {
        {"pootTheToot", `LibName`_getPot, METH_VARARGS, "calls the potential on a single walker"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION > 2

const char `LibName`_doc[] = "`LibName` calculates a potential";
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
    return PyModule_Create(&`LibName`Module);
}

#else

PyMODINIT_FUNC init`LibName`(void)
{
    (void) Py_InitModule("`LibName`", `LibName`Methods);
}

#endif