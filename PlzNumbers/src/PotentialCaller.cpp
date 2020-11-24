

#include "PotentialCaller.hpp"

namespace rynlib {
    namespace PlzNumbers {

        PotValsManager PotentialCaller::get_pot() {
            auto scattered = mpi_manager.scatter_walkers(walker_data);
            auto pots = caller.call_potential(scattered);
            return mpi_manager.gather_potentials(walker_data, pots);
        };
    }
}