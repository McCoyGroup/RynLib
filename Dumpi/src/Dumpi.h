
#ifndef DUMPI_H
#define DUMPI_H

#include "Python.h"
#include "mpi.h"

// We'll do a bunch of typedefs and includes and stuff to make it easier to work with/debug this stuff

/*
 * Python Interface
 *
 * these are the methods that will actually be exported and available to python
 * nothing else will be visible directly, so we need to make sure that this set is sufficient for out purposes
 *
 */

static PyObject *Dumpi_initializeMPI
    ( PyObject *, PyObject * );

static PyObject *Dumpi_finalizeMPI
    ( PyObject *, PyObject * );

static PyObject *Dumpi_syncMPI
        ( PyObject *, PyObject * );

#endif