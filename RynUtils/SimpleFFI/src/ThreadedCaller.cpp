//
// A class to manage the details of actual threading. In its own subclass to enable better extensibility
//

#include "ThreadedCaller.hpp"
#include <algorithm>

#ifdef _OPENMP
#include <omp.h> // comes with -fopenmp
#endif

#ifdef _TBB // I gotta set this now but like it'll allow for better scalability
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#endif

namespace simpleffi {

//        double _doopAPot(
//                Coordinates &walker_coords,
//                Names &atoms,
//                PotentialFunction pot_func,
//                ExtraArgs &args,
//                int retries
//        ) {
//            double pot;
//
//            std::string bad_walkers_file = args.bad_walkers_file;
//            double err_val = args.err_val;
//            bool debug_print = args.debug_print;
//            ExtraBools &extra_bools = args.extra_bools;
//            ExtraInts &extra_ints = args.extra_ints;
//            ExtraFloats &extra_floats = args.extra_floats;
//
//            try {
//                if (debug_print) {
//                    std::string walker_string = _appendWalkerStr("Walker before call: ", "", walker_coords);
//                    printf("%s\n", walker_string.c_str());
//                }
//                pot = pot_func(walker_coords, atoms, extra_bools, extra_ints, extra_floats);
//                if (debug_print) {
//                    printf("  got back energy: %f\n", pot);
//                }
//
//            } catch (std::exception &e) {
//                if (retries > 0) {
//                    return _doopAPot(walker_coords, atoms, pot_func, args, retries - 1);
//                } else {
//                    // pushed error reporting into bad_walkers_file
//                    // should probably revise yet again to print all of this stuff to python's stderr...
//                    if (debug_print) {
//                        std::string no_str = "";
//                        _printOutWalkerStuff(
//                                walker_coords,
//                                no_str,
//                                e.what()
//                        );
//                    } else {
//                        _printOutWalkerStuff(
//                                walker_coords,
//                                bad_walkers_file,
//                                e.what()
//                        );
//                    }
//                    pot = err_val;
//                }
//            }
//
//            return pot;
//        };

//        double _doopAPot(
//                FlatCoordinates &walker_coords,
//                Names &atoms,
//                FlatPotentialFunction pot_func,
//                ExtraArgs &args,
//                int retries
//        ) {
//            double pot;
//
//            std::string bad_walkers_file = args.bad_walkers_file;
//            double err_val = args.err_val;
//            bool debug_print = args.debug_print;
//            ExtraBools &extra_bools = args.extra_bools;
//            ExtraInts &extra_ints = args.extra_ints;
//            ExtraFloats &extra_floats = args.extra_floats;
//
//            try {
//                if (debug_print) {
//                    std::string walker_string = _appendWalkerStr("Walker before call: ", "", walker_coords);
//                    printf("%s\n", walker_string.c_str());
//                }
//                pot = pot_func(walker_coords, atoms, extra_bools, extra_ints, extra_floats);
//
//            } catch (std::exception &e) {
//                if (retries > 0) {
//                    return _doopAPot(walker_coords, atoms, pot_func, args, retries - 1);
//                } else {
//                    if (debug_print) {
//                        std::string no_str = "";
//                        _printOutWalkerStuff(
//                                walker_coords,
//                                no_str,
//                                e.what()
//                        );
//                    } else {
//                        _printOutWalkerStuff(
//                                walker_coords,
//                                bad_walkers_file,
//                                e.what()
//                        );
//                    }
//                    pot = err_val;
//                }
//            }
//
//            return pot;
//        };


//        PotValsManager _vecPotCall(
//                Configurations coords,
//                Names atoms,
//                std::vector<size_t > shape,
//                VectorizedPotentialFunction pot_func,
//                ExtraArgs &args,
//                int retries = 3
//        ) {
//
//            auto debug_print = args.debug_print;
//
//            if (debug_print) {
//                printf("calling vectorized potential on %ld walkers", coords.size());
//            }
//
//            PotValsManager pots;
//            try {
//
//                PotentialVector pot_vec = pot_func(coords,
//                                atoms,
//                                args.extra_bools,
//                                args.extra_ints,
//                                args.extra_floats
//                );
//                pots = PotValsManager(pot_vec, shape[0]);
//
//            } catch (std::exception &e) {
//                if (retries > 0) {
//                    pots = _vecPotCall(coords, atoms, shape, pot_func, args, retries - 1);
//
//                } else {
//                    printf("Error in vectorized call %s\n", e.what());
//                    pots = PotValsManager(shape[0], shape[1], args.err_val);
//                }
//            }
//
//            return pots;
//
//        }

//        PotValsManager _vecPotCall(
//                FlatConfigurations coords,
//                Names atoms,
//                std::vector<size_t > shape,
//                VectorizedFlatPotentialFunction pot_func,
//                ExtraArgs &args,
//                int retries = 3
//        ) {
//
//            auto debug_print = args.debug_print;
//
//            if (debug_print) {
//                printf("calling vectorized potential on %ld walkers", shape[0] * shape[1]);
//            }
//
//            PotValsManager pots;
//            try {
//
//                RawPotentialBuffer pot_dat = pot_func(coords,
//                                atoms,
//                                args.extra_bools,
//                                args.extra_ints,
//                                args.extra_floats
//                );
//                PotentialVector pot_vec(pot_dat, pot_dat + shape[0] * shape[1]);
//                pots = PotValsManager(pot_vec, shape[0]);
//
//            } catch (std::exception &e) {
//                if (retries > 0) {
//                    pots = _vecPotCall(coords, atoms, shape, pot_func, args, retries - 1);
//
//                } else {
//                    printf("Error in vectorized call %s\n", e.what());
//                    pots = PotValsManager(shape[0], shape[1], args.err_val);
//                }
//            }
//
//            return pots;
//
//        }

//        Real_t PotentialApplier::call(
//                CoordsManager &coords,
//                ExtraArgs &args,
//                std::vector<size_t> which
//        ) {
//            Real_t pot_val;
//            if (flat_mode) {
//                auto walker = coords.get_flat_walker(which);
//                auto atoms = coords.get_atoms();
//                pot_val = _doopAPot(
//                        walker,
//                        atoms,
//                        get_func_pointer<FlatPotentialFunction >(),
//                        args,
//                        args.default_retries
//                );
//            } else {
//                auto walker = coords.get_walker(which);
//                auto atoms = coords.get_atoms();
//                pot_val = _doopAPot(
//                        walker,
//                        atoms,
//                        get_func_pointer<PotentialFunction >(),
//                        args,
//                        args.default_retries
//                );
//            }
//            return pot_val;
//        };

        PotValsManager PotentialApplier::call_vectorized(
                CoordsManager &coords,
                ExtraArgs &args
        ) {
            PotValsManager pot_vals;
            if (flat_mode) {
                auto walker = coords.get_flat_walkers();
                auto atoms = coords.get_atoms();
                pot_vals = _vecPotCall(
                        walker,
                        atoms,
                        coords.get_shape(),
                        get_func_pointer<VectorizedFlatPotentialFunction >(),
                        args,
                        args.default_retries
                );
            } else {
                auto walker = coords.get_walkers();
                auto atoms = coords.get_atoms();
                pot_vals = _vecPotCall(
                        walker,
                        atoms,
                        coords.get_shape(),
                        get_func_pointer<VectorizedPotentialFunction >(),
                        args,
                        args.default_retries
                );
            }
            return pot_vals;
        }

        PotValsManager PotentialApplier::call_python(
                CoordsManager &coords,
                ExtraArgs &args
        ) {

            PyObject* coord_obj = coords.as_numpy_array();
            PyObject* py_args = PyTuple_Pack(3, coord_obj, args.py_atoms, args.extra_args);

            PyObject* pot_vals = PyObject_CallObject(py_pot, py_args);
            if (pot_vals == NULL) {
                Py_XDECREF(py_args);
                Py_XDECREF(coord_obj);
                throw std::runtime_error("python issues...");
            }

//            PotValsManager

        }

        PotValsManager ThreadingHandler::call_potential(
                CoordsManager &coords,
                ExtraArgs &args
        ) {
            auto shp = coords.get_shape();
            auto atoms = coords.get_atoms();
            auto ncalls = shp[0];
            auto nwalkers = shp[1];
            PotValsManager pots(ncalls, nwalkers);

            switch (mode) {
                case (ThreadingMode::OpenMP) :
                    ThreadingHandler::_call_omp(pots, coords, args);
                    break;
                case (ThreadingMode::TBB) :
                    ThreadingHandler::_call_tbb(pots, coords, args);
                    break;
                case (ThreadingMode::VECTORIZED) :
                    ThreadingHandler::_call_vec(pots, coords, args);
                    break;
                case (ThreadingMode::PYTHON) :
                    ThreadingHandler::_call_python(pots, coords, args);
                    break;
                case (ThreadingMode::SERIAL) :
                    ThreadingHandler::_call_serial(pots, coords, args);
                default:
                    throw std::runtime_error("Bad threading mode?");
            }

            return pots;
        }

        void _loop_inner(
                PotValsManager &pots,
                CoordsManager &coords,
                PotentialApplier &pot_caller,
                ExtraArgs &args,
                size_t nwalkers,
                size_t w
        ) {
            auto n = (size_t) w / nwalkers;
            auto i = w % nwalkers;

//            RawPotentialBuffer current_data = pots[n].data();

            std::vector<size_t> which{n, i};
            Real_t pot_val = pot_caller.call(
                    coords,
                    args,
                    which
            );

            pots.assign(n, i, pot_val);
        }

        void ThreadingHandler::_call_vec(
                PotValsManager &pots,
                CoordsManager &coords,
                ExtraArgs &args
        ) {
            PotValsManager new_pots = pot.call_vectorized(coords, args);
            pots.assign(new_pots);
        }

        void ThreadingHandler::_call_python(
                PotValsManager &pots,
                CoordsManager &coords,
                ExtraArgs &args
        ) {
            PotValsManager new_pots = pot.call_python(coords, args);
            pots.assign(new_pots);
        }

        void ThreadingHandler::_call_serial(
                PotValsManager &pots,
                CoordsManager &coords,
                ExtraArgs &args
        ) {

            auto shp = coords.get_shape();
            auto atoms = coords.get_atoms();
            auto ncalls = shp[0];
            auto nwalkers = shp[1];

            auto total_walkers = ncalls * nwalkers;
            auto debug_print = args.debug_print;

            for (auto w = 0; w < total_walkers; w++) {
                _loop_inner(
                        pots,
                        coords,
                        pot,
                        args,
                        nwalkers,
                        w
                );
            }
        }

        void ThreadingHandler::_call_omp(
                PotValsManager &pots,
                CoordsManager &coords,
                ExtraArgs &args
        ) {
#ifdef _OPENMP
            auto shp = coords.get_shape();
            auto atoms = coords.get_atoms();
            auto ncalls = shp[0];
            auto nwalkers = shp[1];

            auto total_walkers = ncalls * nwalkers;
            auto debug_print = args.debug_print;

#pragma omp parallel for
            for (auto w = 0; w < total_walkers; w++) {
                _loop_inner(
                        pots,
                        coords,
                        pot,
                        args,
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
                CoordsManager &coords,
                ExtraArgs &args
        ) {
#ifdef _TBB
            auto shp = coords.get_shape();
            auto atoms = coords.get_atoms();
            auto ncalls = shp[0];
            auto nwalkers = shp[1];

            auto total_walkers = ncalls * nwalkers;
            auto debug_print = args.debug_print;

            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, total_walkers),
                    [&](const tbb::blocked_range <size_t> &r) {
                        for (size_t w = r.begin(); w < r.end(); ++w) {
                            _loop_inner(
                                    pots,
                                    coords,
                                    pot,
                                    args,
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