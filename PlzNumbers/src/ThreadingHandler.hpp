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
#include "FFIParameters.hpp"
#include <stdexcept>


namespace rynlib {

    using namespace common;
    using namespace python;
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
            std::string arg_sig = "OOdppppppp";

            std::string bad_walkers_file;
            double err_val;
            bool debug_print;
            int default_retries;

            bool raw_array_pot;
            bool vectorized_potential;
            bool use_openMP;
            bool use_TBB;
            bool python_potential;

            // storage for extra args we might want to pass to functions
            FFIParameters params;

            // python objects to propagate through
            // should never be modified in a multithreaded environment
            // but can safely be messed with in a non-threaded one
            PyObject* py_atoms;
            PyObject* py_params;
            PyObject* extra_args;

            // Pointer to function in capsule
            // will get a real name some day, but today is not that day
            std::string func_ptr_name = "_potential";

        public:

            CallerParameters(PyObject* atoms, PyObject* params) : py_atoms(atoms), py_params(params) {
                init();
            }

            void init() {
                PyObject* bad_walkers_str;
                int passed = PyArg_ParseTuple(
                        py_params,
                        arg_sig.c_str(),
                        extra_args,
                        &bad_walkers_str,
                        &err_val,
                        &debug_print,
                        &default_retries,
                        &raw_array_pot,
                        &vectorized_potential,
                        &use_openMP,
                        &use_TBB,
                        &python_potential
                        );
                if (!passed) {
                    Py_XDECREF(bad_walkers_str);
                    throw std::runtime_error("python args issue?");
                }

                bad_walkers_file = from_python<std::string>(bad_walkers_str);

                Py_XDECREF(bad_walkers_str);

//                PyObject* ext_bool = PyTuple_GetItem(extra_args, 0);
//                PyObject* ext_int = PyTuple_GetItem(extra_args, 1);
//                PyObject* ext_float = PyTuple_GetItem(extra_args, 2);

                params = FFIParameters(extra_args);

//                arg_bools = from_python_iterable<bool>(ext_bool);
//                arg_ints = from_python_iterable<int>(ext_int);
//                arg_floats = from_python_iterable<double>(ext_float);
            };

            std::string bad_walkers_dump() { return bad_walkers_file; }
            bool debug() { return debug_print; }
            double error_val() { return err_val; }

            bool flat_mode() {
                return (python_potential || raw_array_pot);
            }

            std::string func_name() {
                return func_ptr_name;
            }

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

            PyObject* python_atoms() { return py_atoms; };
            PyObject* python_args() { return py_params; };

        };

        class PotentialApplier {
            PyObject* py_pot;
            CallerParameters& params;
            // we need to load the appropriate one these _before_ we start
            // calling from our threads
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
                    params(parameters) {}

            Real_t call(
                    CoordsManager& coords,
                    std::vector<size_t >& which
                    );
            Real_t call(
                    CoordsManager& coords,
                    std::vector<size_t >& which,
                    int retries
            );

            PotValsManager call_vectorized(
                    CoordsManager& coords
            );
            PotValsManager call_vectorized(
                    CoordsManager& coords,
                    int retries
            );

            PotValsManager call_python(
                    CoordsManager& coords
            );

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
                    CoordsManager& coords
                    );

            void _call_omp(
                    PotValsManager &pots,
                    CoordsManager &coords
            );

            void _call_tbb(
                    PotValsManager &pots,
                    CoordsManager &coords
            );

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
