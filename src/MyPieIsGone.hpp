//
// Created by Mark Boyer on 1/29/20.
//

#ifndef RYNLIB_MYPIEISGONE_HPP

#include "RynTypes.hpp"
#include "Python.h"

void _mpiInit(int* world_size, int* world_rank);

void _mpiFinalize();

void _mpiBarrier();

PotentialArray _mpiGetPot(
        RawWalkerBuffer raw_data,
        const Names &atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank
);

#define RYNLIB_MYPIEISGONE_HPP

#endif //RYNLIB_MYPIEISGONE_HPP
