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
            ExtraArgs &extra_args;

        public:

            PotentialCaller(
                    CoordsManager& walkers,
                    MPIManager& mpi,
                    ThreadingHandler& threader,
                    ExtraArgs& args
                    ) :
                    walker_data(walkers),
                    mpi_manager(mpi),
                    caller(threader),
                    extra_args(args)
                    {};

            PotValsManager get_pot();

        };

    }
}

#endif //RYNLIB_POTENTIALCALLER_HPP
