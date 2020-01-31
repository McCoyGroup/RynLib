
#ifndef RYNLIB_H
#define RYNLIB_H

#include "Python.h"

#ifdef SADBOYDEBUG

// Empty do nothing debug config for cleanest debugging

#else

#ifdef IM_A_REAL_BOY
#define I_HAVE_PIE
#define I_AM_A_TREE_PERSON
#endif

#ifdef I_AM_A_TREE_PERSON

#include "dmc_interface.h"
// MillerGroup_entosPotential is really in libentos but this predeclares it

#else

#include <vector>
#include <string>
// for testing we roll our own which always spits out 52
double MillerGroup_entosPotential
        (const std::vector< std::vector<double> > , const std::vector<std::string>, bool hf_only = false) {
    return 52.0;
}

#endif

#ifdef I_HAVE_PIE

#include "mpi.h"

#endif

// We'll do a bunch of typedefs and includes and stuff to make it easier to work with/debug this stuff


#include "RynTypes.hpp"

/*
 * Python Interface
 *
 * these are the methods that will actually be exported and available to python
 * nothing else will be visible directly, so we need to make sure that this set is sufficient for out purposes
 *
 */
static PyObject *RynLib_callPot
    ( PyObject *, PyObject * );

static PyObject *RynLib_callPotVec
    ( PyObject *, PyObject * );

static PyObject *RynLib_testPot
    ( PyObject *, PyObject * );

static PyObject *RynLib_initializeMPI
    ( PyObject *, PyObject * );

static PyObject *RynLib_finalizeMPI
    ( PyObject *, PyObject * );

static PyObject *RynLib_holdMPI
        ( PyObject *, PyObject * );

#endif

#endif