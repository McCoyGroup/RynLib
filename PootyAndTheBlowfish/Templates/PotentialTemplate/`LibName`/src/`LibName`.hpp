//
// Created by Mark Boyer on 1/31/20.
//

#ifndef RYNLIB_LIBNAME_POT_HPP

#include "Python.h"
#include "RynTypes.hpp"

Real_t `LibName`_Potential(const Coordinates coords, const Names atoms);
Real_t `LibNameOfPotential`(const Coordinates coords, const Names atoms);

#define RYNLIB_LIBNAME_POT_HPP

#endif //RYNLIB_LIBNAME_POT_HPP
