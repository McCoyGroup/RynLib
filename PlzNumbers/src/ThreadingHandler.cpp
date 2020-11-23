//
// A class to manage the details of actual threading. In its own subclass to enable better extensibility
//

#include "ThreadingHandler.hpp"
#include <algorithm>

#ifdef _OPENMP
#include <omp.h> // comes with -fopenmp
#endif

#ifdef _TBB // I gotta set this now but like it'll allow for better scalability
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#endif

namespace rynlib {

    using namespace common;
    namespace PlzNumbers {

        // Old callers
        std::string _appendWalkerStr(const char *base_str, const char *msg, Coordinates &walker_coords) {
            std::string walks = base_str;
            walks += msg;
            walks += "(";
            for (size_t i = 0; i < walker_coords.size(); i++) {
                walks += "(";
                for (int j = 0; j < 3; j++) {
                    walks += std::to_string(walker_coords[i][j]);
                    if (j < 2) {
                        walks += ", ";
                    } else {
                        walks += ")";
                    }
                }
                walks += ")";
                if (i < walker_coords.size() - 1) {
                    walks += ", ";
                }
            }
            walks += " )";
            return walks;
        }

        std::string _appendWalkerStr(const char *base_str, const char *msg, FlatCoordinates &walker_coords) {
            std::string err_msg = base_str;
            err_msg += msg;
            err_msg += "(";
            for (size_t i = 0; i < walker_coords.size() / 3; i++) {
                err_msg += "(";
                for (int j = 0; j < 3; j++) {
                    err_msg += std::to_string(walker_coords[i * 3 + j]);
                    if (j < 2) {
                        err_msg += ", ";
                    } else {
                        err_msg += ")";
                    }
                }
                if (i < walker_coords.size() - 1) {
                    err_msg += ", ";
                }
            }
            err_msg += " )";
            return err_msg;
        }

        void _printOutWalkerStuff(
                Coordinates walker_coords,
                const std::string &bad_walkers,
                const char *err_string
        ) {

            std::string err_msg = _appendWalkerStr(err_string, " \n This walker was bad: ( ", walker_coords);

            if (!bad_walkers.empty()) {
                const char *fout = bad_walkers.c_str();
                FILE *err = fopen(fout, "a");
                fprintf(err, "%s\n", err_msg.c_str());
                fclose(err);
            } else {
                printf("%s\n", err_msg.c_str());
            }

        }

        void _printOutWalkerStuff(
                FlatCoordinates walker_coords,
                const std::string &bad_walkers,
                const char *err_string
        ) {

            std::string err_msg = _appendWalkerStr(err_string, " \n This walker was bad: ( ", walker_coords);
            if (!bad_walkers.empty()) {
                const char *fout = bad_walkers.c_str();
                FILE *err = fopen(fout, "a");
                fprintf(err, "%s\n", err_msg.c_str());
                fclose(err);
            } else {
                printf("%s\n", err_msg.c_str());
            }

        }

        void CallerParameters::init() {
            PyObject *bad_walkers_str, *fun_name_str;
            int passed = PyArg_ParseTuple(
                    py_params,
                    arg_sig.c_str(),
                    &caller_api,
                    &fun_name_str,
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

            function_name = from_python<std::string>(fun_name_str);
            Py_XDECREF(fun_name_str);

            bad_walkers_file = from_python<std::string>(bad_walkers_str);
            Py_XDECREF(bad_walkers_str);

            parameters = FFIParameters(extra_args);

            if (caller_api == 1) {
                PyObject* extra_bools = PyTuple_GetItem(extra_args, 0);
                ext_bools = from_python_iterable<bool>(extra_bools);
                PyObject* extra_ints = PyTuple_GetItem(extra_args, 1);
                ext_ints = from_python_iterable<int>(extra_ints);
                PyObject* extra_floats = PyTuple_GetItem(extra_args, 2);
                ext_floats = from_python_iterable<double>(extra_floats);
            }

        };


        // Old API
        Real_t PotentialApplier::call_1(
                CoordsManager &coords,
                std::vector<size_t>& which,
                int retries
        ) {
            Real_t pot_val;

            auto atoms = coords.get_atoms();
            auto bad_walkers_file = params.bad_walkers_dump();
            auto err_val = params.error_val();
            bool debug_print = params.debug();
            auto extra_bools = params.extra_bools();
            auto extra_ints = params.extra_ints();
            auto extra_floats = params.extra_floats();

            try {
                if (debug_print) {
                    std::string walker_string;
                    if (params.flat_mode()) {
                        auto walker = coords.get_flat_walker(which);
                        walker_string = _appendWalkerStr("Walker before call: ", "", walker);
                    } else {
                        auto walker = coords.get_walker(which);
                        walker_string = _appendWalkerStr("Walker before call: ", "", walker);
                    }
                    printf("%s\n", walker_string.c_str());
                }

                if (params.flat_mode()) {
                    auto walker = coords.get_flat_walker(which);
                    pot_val = flat_pot(walker, atoms, extra_bools, extra_ints, extra_floats);
                } else {
                    auto walker = coords.get_walker(which);
                    pot_val = pot(walker, atoms, extra_bools, extra_ints, extra_floats);
                }
//                pot = pot_func(walker_coords, atoms, extra_bools, extra_ints, extra_floats);
                if (debug_print) {
                    printf("  got back energy: %f\n", pot_val);
                }

            } catch (std::exception &e) {
                if (retries > 0) {
                    return call_1(coords, which, retries - 1);
                } else {
                    // pushed error reporting into bad_walkers_file
                    // should probably revise yet again to print all of this stuff to python's stderr...
                    if (debug_print) {
                        bad_walkers_file = "";
                    }
                    if (params.flat_mode()) {
                        auto walker = coords.get_flat_walker(which);
                        _printOutWalkerStuff(walker, bad_walkers_file, e.what());
                    } else {
                        auto walker = coords.get_walker(which);
                        _printOutWalkerStuff(walker, bad_walkers_file, e.what());
                    }
                    pot_val = err_val;
                }
            }

            return pot_val;

        };

        // New API
        Real_t PotentialApplier::call_2(
                CoordsManager &coords,
                std::vector<size_t>& which,
                int retries
        ) {
            Real_t pot_val;

            auto atoms = coords.get_atoms();
            auto bad_walkers_file = params.bad_walkers_dump();
            auto err_val = params.error_val();
            bool debug_print = params.debug();

            auto method = params.get_method();

            try {
                if (debug_print) {
                    std::string walker_string;
                    if (params.flat_mode()) {
                        auto walker = coords.get_flat_walker(which);
                        walker_string = _appendWalkerStr("Walker before call: ", "", walker);
                    } else {
                        auto walker = coords.get_walker(which);
                        walker_string = _appendWalkerStr("Walker before call: ", "", walker);
                    }
                    printf("%s\n", walker_string.c_str());
                }

                auto call_params = params.ffi_params.copy();
                call_params.update_key("coords", coords);
                pot_val = method.call(call_params);
//                pot = pot_func(walker_coords, atoms, extra_bools, extra_ints, extra_floats);
                if (debug_print) {
                    printf("  got back energy: %f\n", pot_val);
                }

            } catch (std::exception &e) {
                if (retries > 0) {
                    return call_2(coords, which, retries - 1);
                } else {
                    // pushed error reporting into bad_walkers_file
                    // should probably revise yet again to print all of this stuff to python's stderr...
                    if (debug_print) {
                        bad_walkers_file = "";
                    }
                    if (params.flat_mode()) {
                        auto walker = coords.get_flat_walker(which);
                        _printOutWalkerStuff(walker, bad_walkers_file, e.what());
                    } else {
                        auto walker = coords.get_walker(which);
                        _printOutWalkerStuff(walker, bad_walkers_file, e.what());
                    }
                    pot_val = err_val;
                }
            }

            return pot_val;

        };

        Real_t PotentialApplier::call(
                CoordsManager &coords,
                std::vector<size_t>& which
        ) {
            switch (params.api_version()) {
                case (1):
                    return call_1(coords, which, params.retries());
                    break;
                case (2):
                    return call_2(coords, which, params.retries());
                    break;
                default:
                    throw std::runtime_error("unkown caller API version");
            }
        }


        PotValsManager PotentialApplier::call_vectorized(
                CoordsManager &coords
        ) {
            {
                switch (params.api_version()) {
                    case (1):
                        return call_vectorized_1(coords, params.retries());
                        break;
                    case (2):
                        return call_vectorized_2(coords, params.retries());
                        break;
                    default:
                        throw std::runtime_error("unkown caller API version");
                }
            }
        }

        PotValsManager PotentialApplier::call_vectorized_1(
                CoordsManager &coords,
                int retries
        ) {
            PotValsManager pot_vals;

            auto debug_print = params.debug();
            auto shape = coords.get_shape();

            if (debug_print) {
                printf("calling vectorized potential on %ld walkers", coords.num_geoms());
            }

            PotValsManager pots;
            try {

                PotentialVector pot_vec;
                if (params.flat_mode()) {
                    RawPotentialBuffer pot_dat = vec_flat_pot(
                            coords.get_flat_walkers(),
                            coords.get_atoms(),
                            params.extra_bools(),
                            params.extra_ints(),
                            params.extra_floats()
                    );
                    pot_vec.assign(
                            pot_dat,
                            pot_dat + coords.num_geoms()
                            );

                } else {
                    pot_vec = vec_pot(
                            coords.get_walkers(),
                            coords.get_atoms(),
                            params.extra_bools(),
                            params.extra_ints(),
                            params.extra_floats()
                    );
                }
                pots = PotValsManager(pot_vec, coords.num_calls());

            } catch (std::exception &e) {
                if (retries > 0) {
                    pots = call_vectorized(coords, retries - 1);
                } else {
                    printf("Error in vectorized call %s\n", e.what());
                    pots = PotValsManager(coords.num_calls(), coords.num_walkers(), params.error_val());
                }
            }

            return pots;
        }

        PotValsManager PotentialApplier::call_python(
                CoordsManager &coords
        ) {

            PyObject* coord_obj = coords.as_numpy_array();
            PyObject* py_args = PyTuple_Pack(3, coord_obj, params.python_atoms(), params.python_args());

            PyObject* pot_vals = PyObject_CallObject(py_pot, py_args);
            if (pot_vals == NULL) {
                Py_XDECREF(py_args);
                Py_XDECREF(coord_obj);
                throw std::runtime_error("python issues...");
            }

            auto ncalls = coords.num_calls();
            auto num_walkers = coords.num_geoms();
            auto data = get_numpy_data<Real_t >(pot_vals);

            PotentialVector pot_vec(data, data+num_walkers);

            return PotValsManager(pot_vec, ncalls);

        }

        PotValsManager ThreadingHandler::call_potential(
                CoordsManager &coords
        ) {
            auto atoms = coords.get_atoms();
            auto ncalls = coords.num_calls();
            auto nwalkers = coords.num_walkers();
            PotValsManager pots(ncalls, nwalkers);

            switch (mode) {
                case (ThreadingMode::OpenMP) :
                    ThreadingHandler::_call_omp(pots, coords);
                    break;
                case (ThreadingMode::TBB) :
                    ThreadingHandler::_call_tbb(pots, coords);
                    break;
                case (ThreadingMode::VECTORIZED) :
                    ThreadingHandler::_call_vec(pots, coords);
                    break;
                case (ThreadingMode::PYTHON) :
                    ThreadingHandler::_call_python(pots, coords);
                    break;
                case (ThreadingMode::SERIAL) :
                    ThreadingHandler::_call_serial(pots, coords);
                default:
                    throw std::runtime_error("Bad threading mode?");
            }

            return pots;
        }

        void _loop_inner(
                PotValsManager &pots,
                CoordsManager &coords,
                PotentialApplier &pot_caller,
                size_t nwalkers,
                size_t w
        ) {
            auto n = (size_t) w / nwalkers;
            auto i = w % nwalkers;

//            RawPotentialBuffer current_data = pots[n].data();

            std::vector<size_t> which{n, i};
            Real_t pot_val = pot_caller.call(
                    coords,
                    which
            );

            pots.assign(n, i, pot_val);
        }

        void ThreadingHandler::_call_vec(
                PotValsManager &pots,
                CoordsManager &coords
        ) {
            PotValsManager new_pots = pot.call_vectorized(coords);
            pots.assign(new_pots);
        }

        void ThreadingHandler::_call_python(
                PotValsManager &pots,
                CoordsManager &coords
        ) {
            PotValsManager new_pots = pot.call_python(coords);
            pots.assign(new_pots);
        }

        void ThreadingHandler::_call_serial(
                PotValsManager &pots,
                CoordsManager &coords
        ) {

            auto atoms = coords.get_atoms();
            auto ncalls = coords.num_calls();
            auto nwalkers = coords.num_walkers();

            auto total_walkers = ncalls * nwalkers;
//            auto debug_print = args.debug_print;

            for (auto w = 0; w < total_walkers; w++) {
                _loop_inner(
                        pots,
                        coords,
                        pot,
                        nwalkers,
                        w
                );
            }
        }

        void ThreadingHandler::_call_omp(
                PotValsManager &pots,
                CoordsManager &coords
        ) {
#ifdef _OPENMP
            auto atoms = coords.get_atoms();
            auto ncalls = coords.num_calls();
            auto nwalkers = coords.num_walkers();

            auto total_walkers = ncalls * nwalkers;
//            auto debug_print = args.debug_print;

#pragma omp parallel for
            for (auto w = 0; w < total_walkers; w++) {
                _loop_inner(
                        pots,
                        coords,
                        pot,
                        nwalkers,
                        w
                );
            }
#else
            throw std::runtime_error("OpenMP not installed");

#endif
        }

        void ThreadingHandler::_call_tbb(
                PotValsManager &pots,
                CoordsManager &coords
        ) {
#ifdef _TBB
            auto atoms = coords.get_atoms();
            auto ncalls = coords.num_calls();
            auto nwalkers = coords.num_walkers();

            auto total_walkers = ncalls * nwalkers;
//            auto debug_print = args.debug_print;

            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, total_walkers),
                    [&](const tbb::blocked_range <size_t> &r) {
                        for (size_t w = r.begin(); w < r.end(); ++w) {
                            _loop_inner(
                                    pots,
                                    coords,
                                    pot,
                                    nwalkers,
                                    w
                            );
                        }
                    }
            );

#else
            throw std::runtime_error("TBB not installed");
#endif
        }

    } // namespace PlzNumbers
}