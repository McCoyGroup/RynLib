
#ifndef PLZNUMBERS_H
#define PLZNUMBERS_H

#include "Python.h"

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

#endif

#endif