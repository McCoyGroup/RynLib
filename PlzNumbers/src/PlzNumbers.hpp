
#ifndef PLZNUMBERS_PYTHONLAYER_HPP
#define PLZNUMBERS_PYTHONLAYER_HPP

#include "Python.h"

// We'll do a bunch of typedefs and includes and stuff to make it easier to work with/debug this stuff

#include "RynTypes.hpp"
#include "PotentialCaller.hpp"

/*
 * Python Interface
 *
 * these are the methods that will actually be exported and available to python
 * nothing else will be visible directly, so we need to make sure that this set is sufficient for out purposes
 *
 */
static PyObject *PlzNumbers_callPotVec
        ( PyObject *, PyObject * );

#endif //PLZNUMBERS_PYTHONLAYER_HPP
