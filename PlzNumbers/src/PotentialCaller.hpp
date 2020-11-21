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

            bool debug_print;
            std::string bad_walkers_file;
            Real_t error_val;

        public:

            PotentialCaller(
                    CoordsManager& walkers,
                    MPIManager& mpi,
                    ThreadingHandler& threader,
                    ExtraArgs& args,
                    bool debug=false,
                    std::string bad_walkers="",
                    Real_t err=1.0e9
                    ) :
                    walker_data(walkers),
                    mpi_manager(mpi),
                    caller(threader),
                    extra_args(args),
                    debug_print(debug),
                    bad_walkers_file(bad_walkers),
                    error_val(err)
                    {};
            PotentialArray get_pot() {};

        };

    }
}

#endif //RYNLIB_POTENTIALCALLER_HPP
