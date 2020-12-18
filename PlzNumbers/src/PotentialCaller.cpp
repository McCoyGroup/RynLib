

#include "PotentialCaller.hpp"

namespace rynlib::PlzNumbers {

        PotValsManager PotentialCaller::get_pot() {
            if (mpi_manager.is_main() && caller.call_parameters().debug()) {
                printf("  > scattering walkers\n");
            }
            auto scattered = mpi_manager.scatter_walkers(walker_data);
            if (mpi_manager.is_main() && caller.call_parameters().debug()) {
                printf("  > calling potential\n");
            }
            auto pots = caller.call_potential(scattered);
            if(!mpi().no_mpi()) {
                if (mpi_manager.is_main() && caller.call_parameters().debug()) {
                    printf("  > cleaning up walkers\n");
                }
                scattered.cleanup();
            }
            if (mpi_manager.is_main() && caller.call_parameters().debug()) {
                printf("  > gathering potentials\n");
            }
            return mpi_manager.gather_potentials(scattered, pots);
        };
}