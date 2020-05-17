
#ifndef DoMyCode_H
#define DoMyCode_H

#include "Python.h"
#include "RynTypes.hpp"

/*
 * Python Interface
 *
 * these are the methods that will actually be exported and available to python
 * nothing else will be visible directly, so we need to make sure that this set is sufficient for out purposes
 *
 */
namespace DoMyCode {

    static PyObject *DoMyCode_distributeWalkers
            ( PyObject *, PyObject * );
    static PyObject *DoMyCode_getWalkersAndPots
            ( PyObject *, PyObject * );
    //static PyObject *DoMyCode_branchWalkers
    //        ( PyObject *, PyObject * );
}

#endif