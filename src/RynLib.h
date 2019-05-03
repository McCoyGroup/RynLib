
#include "Python.h"
#include <vector>
#include <string>
#include "dmc_interface.h"
//#include "mpi.h" // need to load MPI

extern "C" {

static PyObject *RynLib_callPot
    ( PyObject *, PyObject * );

static PyObject *RynLib_callPotVec
    ( PyObject *, PyObject * );

static PyObject *RynLib_testPot
    ( PyObject *, PyObject * );

}

