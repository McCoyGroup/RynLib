
#ifndef PLZNUMBERS_H
#define PLZNUMBERS_H

#include "Python.h"

// We'll do a bunch of typedefs and includes and stuff to make it easier to work with/debug this stuff

#include "RynTypes.hpp"

/*
 * Python Interface
 *
 * these are the methods that will actually be exported and available to python
 * nothing else will be visible directly, so we need to make sure that this set is sufficient for out purposes
 *
 */
static PyObject *PlzNumbers_callPot
        ( PyObject *, PyObject * );

static PyObject *PlzNumbers_callPotVec
        ( PyObject *, PyObject * );

static PyObject *PlzNumbers_callPyPotVec
        ( PyObject *, PyObject * );

#endif
