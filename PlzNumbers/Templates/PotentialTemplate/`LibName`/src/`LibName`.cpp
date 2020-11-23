
#include "`LibName`.hpp"
#include "FFIParameters.hpp"
#include "FFIModules.hpp"

rynlib::PlzNumbers::FFIModule module(
        "`LibName`",
        "`LibDoc`"
        );

`MethodWrappers`

static auto `LibName`Module = module::def(); // the static here is important for memory I think...
PyMODINIT_FUNC PyInit_`LibName`(void)
{
    PyObject *m;
    m = PyModule_Create(&`LibName`Module);
    if (m == NULL) { return NULL; }
    if (!module::attach(m)) { return NULL; }

    return m;
}