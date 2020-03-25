#ifndef RYNLIB_`LIBNAME`_POT_HPP

#include "Python.h"
#include "RynTypes.hpp"

Real_t `LibName`_Potential(
    const Coordinates,
    const Names,
    const ExtraBools,
    const ExtraInts,
    const ExtraFloats
    );
Real_t `PotentialCallDeclaration`;

#define RYNLIB_`LIBNAME`_POT_HPP

#endif //RYNLIB_`LIBNAME`_POT_HPP
