//
//
//

#ifndef RYNLIB_THREADINGHANDLER_HPP
#define RYNLIB_THREADINGHANDLER_HPP

//#include "PotentialCaller.hpp"
#include "RynTypes.hpp"
#include "CoordsManager.hpp"

namespace rynlib {
    namespace PlzNumbers {

        struct ExtraArgs {
            // storage for extra args we might want to pass to functions
            std::string &bad_walkers_file;
            double err_val;
            bool debug_print;
            int default_retries;

            ExtraBools &extra_bools;
            ExtraInts &extra_ints;
            ExtraFloats &extra_floats;
        };

        enum class ThreadingMode {
            OpenMP,
            TBB,
            SERIAL,
            VECTORIZED
        };

        class PotentialApplier {
            PotentialFunction pot;
            FlatPotentialFunction flat_pot;
            bool flat_mode;
        public:
            PotentialApplier(
                    PotentialFunction pot_func,
                    FlatPotentialFunction flat_pot_func
            ) :
                    pot(pot_func),
                    flat_pot(flat_pot_func) {
                flat_mode = (pot == NULL); // We can get this from python
            };

            Real_t call(
                    CoordsManager& coords,
                    ExtraArgs& extraArgs,
                    std::vector<size_t > which
                    );

            PotentialArray call_vectorized(
                    CoordsManager& coords,
                    ExtraArgs& extraArgs
            );


        };

        class ThreadingHandler {
            ThreadingMode mode;
        public:
            PotentialArray call_potential(
                    CoordsManager& coords,
                    PotentialApplier& pot,
                    ExtraArgs& extraArgs
                    );

            void _call_omp(
                    PotentialArray &pots,
                    CoordsManager &coords,
                    PotentialApplier& pot,
                    ExtraArgs &args
            );

            void _call_tbb(
                    PotentialArray &pots,
                    CoordsManager &coords,
                    PotentialApplier& pot,
                    ExtraArgs &args
            );

            void _call_vec(
                    PotentialArray &pots,
                    CoordsManager &coords,
                    PotentialApplier& pot,
                    ExtraArgs &args
            );

            void _call_serial(
                    PotentialArray &pots,
                    CoordsManager &coords,
                    PotentialApplier& pot,
                    ExtraArgs &args
            );

        };
    }
}


#endif //RYNLIB_THREADINGHANDLER_HPP
