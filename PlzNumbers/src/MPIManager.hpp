//
// C++ side layer to work with the MPI-manager object exposed by Dumpi
//

#ifndef RYNLIB_MPIMANAGER_HPP
#define RYNLIB_MPIMANAGER_HPP

#include "RynTypes.hpp"
#include "CoordsManager.hpp"
#include "PotValsManager.hpp"

using namespace rynlib::common;

// these names should be rescoped probably...
typedef int (*ScatterFunction)(PyObject*, RawWalkerBuffer, int, int, RawWalkerBuffer);
//typedef int (*GatherWalkerFunction)(PyObject*, RawWalkerBuffer, int, int, RawWalkerBuffer);
typedef int (*GatherFunction)(PyObject*, RawPotentialBuffer, int, RawPotentialBuffer);

namespace rynlib {

    namespace PlzNumbers {
        class MPIManager {
            PyObject *mpi_manager;
            int mpi_world_size = -1;
            int mpi_world_rank = -1;
        public:

            // CLion told me I had to add the explicit ?_?
            explicit MPIManager(PyObject *m) : mpi_manager(m) {};
            CoordsManager scatter_walkers(CoordsManager& coords);
//            CoordsManager gather_walkers(CoordsManager& coords);
            PotValsManager gather_potentials(  CoordsManager& coords, PotValsManager& pots );

            void init();
            bool no_mpi() {
                return (mpi_manager == Py_None);
            }
            int world_size() {
                if (mpi_world_size == -1) { init(); }
                return mpi_world_size;
            }
            int world_rank() {
                if (mpi_world_rank == -1) { init(); }
                return mpi_world_rank;
            }
            bool is_main() {
                return world_rank() == 0;
            }
        };
    }
}
#endif //RYNLIB_MPIMANAGER_HPP
