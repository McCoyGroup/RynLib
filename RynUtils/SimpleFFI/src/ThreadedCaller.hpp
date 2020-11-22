//
//
//

#ifndef RYNLIB_THREADINGHANDLER_HPP
#define RYNLIB_THREADINGHANDLER_HPP

//#include "PotentialCaller.hpp"
#include "SimpleFFI.hpp"
#include "PyAllUp.hpp"
#include <exception>


namespace simpleffi {

    using namespace common;
    using namespace python;
    namespace PlzNumbers {

        class ExtraArgs { // The goal is for this to basically directly parallel the python-side "extra args"
            // flags to be fed to the code
            std::string arg_sig = "OdppppOppp";
            std::string &bad_walkers_file;
            double err_val;
            bool debug_print;
            int default_retries;

            // storage for extra args we might want to pass to functions
            ExtraBools &extra_bools;
            ExtraInts &extra_ints;
            ExtraFloats &extra_floats;

            // python objects to propagate through
            // should never be modified in a multithreaded environment
            // but can safely be messed with in a non-threaded one
            PyObject* py_atoms;
            PyObject* extra_args;
        };

        enum class ThreadingMode {
            OpenMP,
            TBB,
            SERIAL,
            VECTORIZED,
            PYTHON
        };

        class PotentialApplier {
            PyObject* py_pot;
            ThreadingMode mode;
            bool flat_mode;
            std::string pointer_name;
        public:
            PotentialApplier(
                    PyObject* python_pot,
                    ThreadingMode threading_mode,
                    bool raw_array_potential,
                    std::string func = "_potential"
            ) :
                    py_pot(python_pot),
                    mode(threading_mode),
                    flat_mode(raw_array_potential),
                    pointer_name(func)
            {}

            Real_t call(
                    CoordsManager& coords,
                    ExtraArgs& extraArgs,
                    std::vector<size_t > which
                    );

            PotValsManager call_vectorized(
                    CoordsManager& coords,
                    ExtraArgs& extraArgs
            );

            bool flat_caller() { return (flat_mode || (mode == ThreadingMode::PYTHON)); }

            template<typename T>
            T get_func_pointer() {
                return from_python_capsule<T>(py_pot, pointer_name.c_str());
            }

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

            void _call_python(
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
