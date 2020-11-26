//
//
//

#ifndef RYNLIB_THREADINGHANDLER_HPP
#define RYNLIB_THREADINGHANDLER_HPP

//#include "PotentialCaller.hpp"
#include "RynTypes.hpp"
#include "CoordsManager.hpp"
#include "PotValsManager.hpp"
#include "PyAllUp.hpp"
#include "plzffi/FFIParameters.hpp"
#include "plzffi/FFIModule.hpp"
#include <stdexcept>

namespace rynlib {

    using namespace common;
    using namespace python;
    using namespace plzffi;

    namespace PlzNumbers {

        enum class ThreadingMode {
            OpenMP,
            TBB,
            SERIAL,
            VECTORIZED,
            PYTHON
        };

        class CallerParameters { // The goal is for this to basically directly parallel the python-side "extra args"

            // flags to be fed to the code
            std::string arg_sig = "iOOOdpippppp";

            int caller_api = 0;
            // To support both old style and new style potential calls by
            // inferring from the name of the module parameters
            std::string function_name; // = "_potential";
            // To be able to recieve a function name from the python side

            PyObject* extra_args = NULL;
            std::string bad_walkers_file;
            double err_val = 1.0e9;
            bool debug_print = false;
            int default_retries = 1;

            bool raw_array_pot = false;
            bool vectorized_potential = false;
            bool use_openMP = true;
            bool use_TBB = false;
            bool python_potential = false;

            // used for the old-style caller
            ExtraBools ext_bools = {};
            ExtraInts ext_ints = {};
            ExtraFloats ext_floats = {};

            // storage for extra args we might want to pass to functions
            FFIParameters parameters;

            // python objects to propagate through
            // should never be modified in a multithreaded environment
            // but can safely be messed with in a non-threaded one
            PyObject* py_atoms;
            PyObject* py_params;

        public:

            CallerParameters(PyObject* atoms, PyObject* pars) : py_atoms(atoms), py_params(pars) {
                init();
            }

            void init();

            std::string bad_walkers_dump() { return bad_walkers_file; }
            bool debug() { return debug_print; }
            double error_val() { return err_val; }

            bool flat_mode() {
                return (python_potential || raw_array_pot);
            }

            std::string func_name() { return function_name; }

            ThreadingMode threading_mode() {
                if (python_potential) {
                    return ThreadingMode::PYTHON;
                } else if (vectorized_potential) {
                    return ThreadingMode::VECTORIZED;
                } else if (use_openMP) {
                    return ThreadingMode::OpenMP;
                } else if (use_TBB) {
                    return ThreadingMode::TBB;
                } else {
                    return ThreadingMode::SERIAL;
                }
            }

            int retries() { return default_retries; }

            PyObject* python_atoms() { return py_atoms; }
            PyObject* python_args() { return py_params; }

            ExtraBools extra_bools() { return ext_bools; }
            ExtraInts extra_ints() { return ext_ints; }
            ExtraFloats extra_floats() { return ext_floats; }

            int api_version() { return caller_api; }

            FFIParameters ffi_params() { return parameters; }

        };

        class PotentialApplier {
            PyObject* py_pot;
            CallerParameters params;
            // we need to load the appropriate one of
            // these _before_ we start calling from our threads
            FFIModule module;
            PotentialFunction pot;
            FlatPotentialFunction flat_pot;
            VectorizedPotentialFunction vec_pot;
            VectorizedFlatPotentialFunction vec_flat_pot;
        public:
            PotentialApplier(
                    PyObject* python_pot,
                    CallerParameters& parameters
            ) :
                    py_pot(python_pot),
                    params(parameters) {
                init();
            }

            void init();

            template<typename T>
            FFIMethod<T> get_method();

            CallerParameters call_parameters() {

//                printf("....????? %d %s\n", params.api_version(), "bloop");//get_python_repr(params.python_args()).c_str() );
//                printf("   ????? wtf?\n" );
                return params;
            }

            Real_t call(CoordsManager& coords, std::vector<size_t >& which);
            Real_t call_1(CoordsManager& coords, std::vector<size_t >& which, int retries);
            Real_t call_2(CoordsManager& coords, std::vector<size_t >& which, int retries);

            PotValsManager call_vectorized(CoordsManager& coords);
            PotValsManager call_vectorized_1(CoordsManager& coords, int retries);
            PotValsManager call_vectorized_2(CoordsManager& coords, int retries);

            PotValsManager call_python(CoordsManager& coords);

        };

        class ThreadingHandler {
            PotentialApplier pot;
            ThreadingMode mode;
        public:
            ThreadingHandler(PotentialApplier& pot_func, ThreadingMode threading) : pot(pot_func), mode(threading) {}

            PotValsManager call_potential(CoordsManager& coords);

            CallerParameters call_parameters() {auto wat = pot.call_parameters(); return wat;}

            void _call_omp(PotValsManager &pots, CoordsManager &coords);

            void _call_tbb(PotValsManager &pots, CoordsManager &coords);

            void _call_vec(
                    PotValsManager &pots,
                    CoordsManager &coords
            );

            void _call_python(
                    PotValsManager &pots,
                    CoordsManager &coords
            );

            void _call_serial(
                    PotValsManager &pots,
                    CoordsManager &coords
            );

        };
    }
}


#endif //RYNLIB_THREADINGHANDLER_HPP
