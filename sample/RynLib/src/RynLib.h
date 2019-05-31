
#include "Python.h"
#include <vector>
#include <string>

#ifdef IM_A_REAL_BOY

#include "dmc_interface.h"
#include "mpi.h"
// MillerGroup_entosPotential is really in libentos but this predeclares it

#else
// for testing we roll our own which always spits out 52
double MillerGroup_entosPotential(
        const std::vector< std::vector<double> > ,
        const std::vector<std::string>,
        bool hf_only = false
        ){

    return 52.0;

}
#endif //ENTOS_ML_DMC_INTERFACE_H

static PyObject *RynLib_callPot
    ( PyObject *, PyObject * );

static PyObject *RynLib_callPotVec
    ( PyObject *, PyObject * );

static PyObject *RynLib_testPot
    ( PyObject *, PyObject * );
