//
// Created by Mark Boyer on 1/30/20.
//

#ifndef RYNLIB_POTATORS_HPP

#include "RynTypes.hpp"
#include "Python.h"
#include <stdio.h>

void _printOutWalkerStuff( Coordinates walker_coords );

double _doopAPot(const Coordinates &walker_coords, const Names &atoms);

inline int ind2d(int i, int j, int n, int m);
inline int int3d(int i, int j, int k, int m, int l);
Coordinates _getWalkerCoords(const double* raw_data, int i, Py_ssize_t num_atoms);

inline int int4d(int i, int j, int k, int a, int n, int m, int l, int o);
Coordinates _getWalkerCoords2(const double* raw_data, int n, int i, int ncalls, int num_walkers, Py_ssize_t num_atoms);

#define RYNLIB_POTATORS_HPP

#endif //RYNLIB_POTATORS_HPP
