//
// Created by Mark Boyer on 1/29/20.
//

#include "Python.h"
#include "RynTypes.hpp"
#include "Potators.hpp"
#include "mpi.h"

void _mpiInit(int* world_size, int* world_rank) {
    *world_size = 1;
    *world_rank = 0;
}

void _mpiFinalize() {
    // boop
}

void _mpiBarrier() {
    // boop
}

PotentialArray _mpiGetPot(
        double* raw_data,
        Names atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank
) {

    PotentialArray potVals(num_walkers, PotentialVector(ncalls, 0));
    for (int n = 0; n < ncalls; n++) {
        for (int i = 0; i < num_walkers; i++) {
            potVals[i][n] = _doopAPot(_getWalkerCoords2(raw_data, n, i, ncalls, num_walkers, num_atoms), atoms);
        }
    }
    return potVals;

}

