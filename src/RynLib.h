
#include "Python.h"
#include <vector>
#include <string>

#ifdef IM_A_REAL_BOY
#include "dmc_interface.h" // TURN THIS BACK ON TO ACTUALLY USE THIS; OFF FOR TESTING PURPOSES
//#include "mpi.h" // need to load MPI
#endif

extern "C" {

static PyObject *RynLib_callPot
    ( PyObject *, PyObject * );

static PyObject *RynLib_callPotVec
    ( PyObject *, PyObject * );

static PyObject *RynLib_testPot
    ( PyObject *, PyObject * );

}

