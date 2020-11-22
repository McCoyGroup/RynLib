//
// Light-weight potential caller that can thread or call serially or in a vectorized manner
//

#ifndef RYNLIB_POTENTIALCALLER_HPP
#define RYNLIB_POTENTIALCALLER_HPP

#include "RynTypes.hpp"
#include "CoordsManager.hpp"
#include "MPIManager.hpp"
#include "ThreadingHandler.hpp"

namespace rynlib {
    namespace PlzNumbers {

        class PotentialCaller {

            CoordsManager &walker_data;
            MPIManager &mpi_manager;
            ThreadingHandler &caller;

        public:

            PotentialCaller(
                    CoordsManager& walkers,
                    MPIManager& mpi,
                    ThreadingHandler& threader
                    ) :
                    walker_data(walkers),
                    mpi_manager(mpi),
                    caller(threader)
                    {};

            PotValsManager get_pot();

        };

    }
}

#endif //RYNLIB_POTENTIALCALLER_HPP
