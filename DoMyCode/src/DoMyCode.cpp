#include "DoMyCode.h"

#include "WalkerPropagator.h"

// New design that will make use of the WalkerPropagator object I'm setting up
PyObject *DoMyCode_getWalkers( PyObject* self, PyObject* args ) {

    PyObject* cores;
    if ( !PyArg_ParseTuple(args, "O", &cores) ) return NULL;

    Coordinates walker_positions = _mpiGetWalkersFromNodes();

}

// PYTHON WRAPPER EXPORT

static PyMethodDef DoMyCodeMethods[] = {
    {"walkyTalky", DoMyCode_getWalkers, METH_VARARGS, "gets Walkers in a WalkerPropagator env"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION > 2

const char DoMyCode_doc[] = "DoMyCode does some DMC";
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