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

        void _sigillHandler(int signum) {
            printf("Illegal instruction signal (%d) received.\n", signum);
            abort();
//    exit(signum);
        }

        void _sigsevHandler(int signum) {
            printf("Segfault signal (%d) received.\n", signum);
            abort();
        }

        double _doopAPot(
                Coordinates &walker_coords,
                Names &atoms,
                PotentialFunction pot_func,
                ExtraArgs &args,
                int retries
        ) {
            double pot;

            std::string bad_walkers_file = args.bad_walkers_file;
            double err_val = args.err_val;
            bool debug_print = args.debug_print;
            ExtraBools &extra_bools = args.extra_bools;
            ExtraInts &extra_ints = args.extra_ints;
            ExtraFloats &extra_floats = args.extra_floats;

            try {
                if (debug_print) {
                    std::string walker_string = _appendWalkerStr("Walker before call: ", "", walker_coords);
                    printf("%s\n", walker_string.c_str());
                }
                pot = pot_func(walker_coords, atoms, extra_bools, extra_ints, extra_floats);
                if (debug_print) {
                    printf("  got back energy: %f\n", pot);
                }

            } catch (std::exception &e) {
                if (retries > 0) {
                    return _doopAPot(walker_coords, atoms, pot_func, args, retries - 1);
                } else {
                    // pushed error reporting into bad_walkers_file
                    // should probably revise yet again to print all of this stuff to python's stderr...
                    if (debug_print) {
                        std::string no_str = "";
                        _printOutWalkerStuff(
                                walker_coords,
                                no_str,
                                e.what()
                        );
                    } else {
                        _printOutWalkerStuff(
                                walker_coords,
                                bad_walkers_file,
                                e.what()
                        );
                    }
                    pot = err_val;
                }
            }

            return pot;
        };

        double _doopAPot(
                FlatCoordinates &walker_coords,
                Names &atoms,
                FlatPotentialFunction pot_func,
                ExtraArgs &args,
                int retries
        ) {
            double pot;

            std::string bad_walkers_file = args.bad_walkers_file;
            double err_val = args.err_val;
            bool debug_print = args.debug_print;
            ExtraBools &extra_bools = args.extra_bools;
            ExtraInts &extra_ints = args.extra_ints;
            ExtraFloats &extra_floats = args.extra_floats;

            try {
                if (debug_print) {
                    std::string walker_string = _appendWalkerStr("Walker before call: ", "", walker_coords);
                    printf("%s\n", walker_string.c_str());
                }
                pot = pot_func(walker_coords, atoms, extra_bools, extra_ints, extra_floats);

            } catch (std::exception &e) {
                if (retries > 0) {
                    return _doopAPot(walker_coords, atoms, pot_func, args, retries - 1);
                } else {
                    if (debug_print) {
                        std::string no_str = "";
                        _printOutWalkerStuff(
                                walker_coords,
                                no_str,
                                e.what()
                        );
                    } else {
                        _printOutWalkerStuff(
                                walker_coords,
                                bad_walkers_file,
                                e.what()
                        );
                    }
                    pot = err_val;
                }
            }

            return pot;
        };


        PotValsManager _vecPotCall(
                Configurations coords,
                Names atoms,
                std::vector<size_t > shape,
                VectorizedPotentialFunction pot_func,
                ExtraArgs &args,
                int retries = 3
        ) {

            auto debug_print = args.debug_print;

            if (debug_print) {
                printf("calling vectorized potential on %ld walkers", coords.size());
            }

            PotValsManager pots;
            try {

                PotentialVector pot_vec = pot_func(coords,
                                atoms,
                                args.extra_bools,
                                args.extra_ints,
                                args.extra_floats
                );
                pots = PotValsManager(pot_vec, shape[0]);

            } catch (std::exception &e) {
                if (retries > 0) {
                    pots = _vecPotCall(coords, atoms, shape, pot_func, args, retries - 1);

                } else {
                    printf("Error in vectorized call %s\n", e.what());
                    pots = PotValsManager(shape[0], shape[1], args.err_val);
                }
            }

            return pots;

        }

        PotValsManager _vecPotCall(
                FlatConfigurations coords,
                Names atoms,
                std::vector<size_t > shape,
                VectorizedFlatPotentialFunction pot_func,
                ExtraArgs &args,
                int retries = 3
        ) {

            auto debug_print = args.debug_print;

            if (debug_print) {
                printf("calling vectorized potential on %ld walkers", shape[0] * shape[1]);
            }

            PotValsManager pots;
            try {

                RawPotentialBuffer pot_dat = pot_func(coords,
                                atoms,
                                args.extra_bools,
                                args.extra_ints,
                                args.extra_floats
                );
                PotentialVector pot_vec(pot_dat, pot_dat + shape[0] * shape[1]);
                pots = PotValsManager(pot_vec, shape[0]);

            } catch (std::exception &e) {
                if (retries > 0) {
                    pots = _vecPotCall(coords, atoms, shape, pot_func, args, retries - 1);

                } else {
                    printf("Error in vectorized call %s\n", e.what());
                    pots = PotValsManager(shape[0], shape[1], args.err_val);
                }
            }

            return pots;

        }

        Real_t PotentialApplier::call(
                CoordsManager &coords,
                ExtraArgs &args,
                std::vector<size_t> which
        ) {
            Real_t pot_val;
            if (flat_mode) {
                auto walker = coords.get_flat_walker(which);
                auto atoms = coords.get_atoms();
                pot_val = _doopAPot(
                        walker,
                        atoms,
                        flat_pot,
                        args,
                        args.default_retries
                );
            } else {
                auto walker = coords.get_walker(which);
                auto atoms = coords.get_atoms();
                pot_val = _doopAPot(
                        walker,
                        atoms,
                        pot,
                        args,
                        args.default_retries
                );
            }
            return pot_val;
        };

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
                        v_flat_pot,
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
                        v_pot,
                        args,
                        args.default_retries
                );
            }
            return pot_vals;
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

            if (mode == ThreadingMode::OpenMP) {
                ThreadingHandler::_call_omp(pots, coords, args);
            } else if (mode == ThreadingMode::TBB) {
                ThreadingHandler::_call_tbb(pots, coords, args);
            } else if (mode == ThreadingMode::VECTORIZED) {
                ThreadingHandler::_call_vec(pots, coords, args);
            } else {
                ThreadingHandler::_call_serial(pots, coords, args);
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