//
// C++ side layer to work with the MPI-manager object exposed by Dumpi
//

#ifndef RYNLIB_MPIMANAGER_HPP
#define RYNLIB_MPIMANAGER_HPP

#include "RynTypes.hpp"
#include "CoordsManager.cpp"

typedef int (*ScatterFunction)(PyObject*, RawWalkerBuffer, int, int, RawWalkerBuffer);
typedef int (*GatherWalkerFunction)(PyObject*, RawWalkerBuffer, int, int, RawWalkerBuffer);
typedef int (*GatherFunction)(PyObject*, RawPotentialBuffer, int, RawPotentialBuffer);

class MPIManager {
    PyObject* mpi_manager;
public:
    MPIManager(PyObject* m) : mpi_manager(m) {};
    CoordsManager scatter_walkers();
    CoordsManager gather_walkers();
    PotentialArray gather_potentials();
};

#endif //RYNLIB_MPIMANAGER_HPP
