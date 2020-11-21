//
//
//

#ifndef RYNLIB_THREADINGHANDLER_HPP
#define RYNLIB_THREADINGHANDLER_HPP

//#include "PotentialCaller.hpp"
#include "RynTypes.hpp"
#include "CoordsManager.hpp"
#include "PotValsManager.hpp"


namespace rynlib {

    using namespace common;
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
            VectorizedPotentialFunction v_pot;
            VectorizedFlatPotentialFunction v_flat_pot;
            bool flat_mode;
            bool vec_mode;
        public:
            PotentialApplier(
                    PotentialFunction pot_func,
                    FlatPotentialFunction flat_pot_func,
                    VectorizedPotentialFunction vec_pot_func,
                    VectorizedFlatPotentialFunction vec_flat_pot_func
            ) :
                    pot(pot_func),
                    flat_pot(flat_pot_func),
                    v_pot(vec_pot_func),
                    v_flat_pot(vec_flat_pot_func)
                     {
                if (pot != NULL) { // from Python
                    flat_mode = false;
                    vec_mode = false;
                } else if (flat_pot != NULL)  {
                    flat_mode = true;
                    vec_mode = false;
                } else if (v_pot != NULL)  {
                    flat_mode = false;
                    vec_mode = true;
                } else if (v_flat_pot != NULL)  {
                    flat_mode = true;
                    vec_mode = true;
                } else {
                    throw std::logic_error("wat...no potential????");
                }

            };

            Real_t call(
                    CoordsManager& coords,
                    ExtraArgs& extraArgs,
                    std::vector<size_t > which
                    );

            PotValsManager call_vectorized(
                    CoordsManager& coords,
                    ExtraArgs& extraArgs
            );

            bool flat_caller() { return flat_mode; }

        };

        class ThreadingHandler {
            PotentialApplier& pot;
            ThreadingMode mode;
        public:
            ThreadingHandler(
                    PotentialApplier& pot_func,
                    ThreadingMode threading
                    ) : pot(pot_func), mode(threading) {}

            PotValsManager call_potential(
                    CoordsManager& coords,
                    ExtraArgs& extraArgs
                    );

            void _call_omp(
                    PotValsManager &pots,
                    CoordsManager &coords,
                    ExtraArgs &args
            );

            void _call_tbb(
                    PotValsManager &pots,
                    CoordsManager &coords,
                    ExtraArgs &args
            );

            void _call_vec(
                    PotValsManager &pots,
                    CoordsManager &coords,
                    ExtraArgs &args
            );

            void _call_serial(
                    PotValsManager &pots,
                    CoordsManager &coords,
                    ExtraArgs &args
            );

        };
    }
}


#endif //RYNLIB_THREADINGHANDLER_HPP
