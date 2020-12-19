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
#include "tbb/task_arena.h"
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
                for (size_t j = 0; j < 3; j++) {
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
                for (size_t j = 0; j < 3; j++) {
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

//            printf("  > . > . > 1\n");

            if (rynlib::common::debug_print()) {
                printf("  > getting argument signature...\n");
            }
            std::string true_arg_sig = get_python_attr<std::string>(py_params, "arg_sig");
            if (true_arg_sig != arg_sig) {
                std::string err = "CallerParameters: argument signature '"
                                  + true_arg_sig
                                  + "' doesn't match expected signature " + arg_sig;
                PyErr_SetString(
                        PyExc_ValueError,
                        err.c_str()
                        );
                throw std::runtime_error("bad python shit");
            }

            if (rynlib::common::debug_print()) {
                printf("  > getting argvec...\n");
            }
            auto argvec = get_python_attr<PyObject *>(py_params, "argvec");

            if (rynlib::common::debug_print()) {
                auto garb = get_python_repr(argvec);
                printf("  > got %s...\n", garb.c_str());
            }

            int dbprint, retries, raw_pot, vecced, useOMP, useTBB, pyPot;
            PyObject *bad_walkers_str, *fun_name_str;
            int passed = PyArg_ParseTuple(
                    argvec,
                    arg_sig.c_str(),
                    &caller_api,
                    &fun_name_str,
                    &extra_args,
                    &bad_walkers_str,
                    &err_val,
                    &dbprint,
                    &retries,
                    &raw_pot,
                    &vecced,
                    &useOMP,
                    &useTBB,
                    &pyPot
            );

//            printf("  >>> 2.2 %f\n", err_val);

            if (!passed) {
                Py_XDECREF(bad_walkers_str);
                Py_XDECREF(fun_name_str);
                throw std::runtime_error("python args issue?");
            }

//            printf("  >>> 3\n");

            debug_print = dbprint;
            default_retries = retries;
            raw_array_pot = raw_pot;
            vectorized_potential = vecced;
            use_openMP = useOMP;
            use_TBB = useTBB;
            python_potential = pyPot;

            rynlib::common::set_debug_print(debug_print);

            function_name = from_python<std::string>(fun_name_str);
            Py_XDECREF(fun_name_str);

            if (rynlib::common::debug_print()) {
                printf("  > loading caller for %s...\n", function_name.c_str());
            }

//            printf("  >>> 4 %s\n", function_name.c_str());

            bad_walkers_file = from_python<std::string>(bad_walkers_str);
            Py_XDECREF(bad_walkers_str);

            if (rynlib::common::debug_print()) {
                printf("  > writing errors to %s...\n", (bad_walkers_file.empty()) ? "stdout" : bad_walkers_file.c_str());
            }

            if (caller_api < 1 || caller_api > 2) {
                Py_XDECREF(bad_walkers_str);
                Py_XDECREF(fun_name_str);
//                printf("  >>> 4 %d\n", caller_api);
                if (rynlib::common::debug_print()) {
                    printf("  > ERROR: bad API version %d...\n", caller_api);
                }
                throw std::runtime_error("Bad API version");
            }
//            printf("  >>> 5\n");


            switch (caller_api) {
                case 1: {
                    if (rynlib::common::debug_print()) printf("  > using old caller API...\n");
                    if (rynlib::common::debug_print()) printf("  > getting extra bools...\n");
                    ext_bools = get_python_attr_iterable<bool>(extra_args, "extra_bools");
                    if (rynlib::common::debug_print()) printf("  > getting extra ints...\n");
                    ext_ints = get_python_attr_iterable<int>(extra_args, "extra_ints");
                    if (rynlib::common::debug_print()) printf("  > getting extra floats...\n");
                    ext_floats = get_python_attr_iterable<double>(extra_args, "extra_floats");
                    break;
                }
                case 2: {
                    if (rynlib::common::debug_print()) printf("  > using new caller API...\n");
                    if (rynlib::common::debug_print()) {
                        auto garb = get_python_repr(extra_args);
                        printf("  > loading parameters from %s...\n", garb.c_str());
                    }
                    parameters = FFIParameters(extra_args);
                    break;
                }
                default:
                    throw std::runtime_error("unkown caller api version");
            }
        };

        void PotentialApplier::init() {
            switch(params.api_version()) {
                case (1): {
                    switch (params.threading_mode()) {
                        case ThreadingMode::PYTHON:
                            break;
                        case ThreadingMode::VECTORIZED: {
                            if (params.flat_mode()) {
                                if (debug_print()) printf("  > getting pointer to vectorized flat potential %s (pointer: %s)\n",
                                                          get_python_repr(py_pot).c_str(),
                                                          params.func_name().c_str());
                                vec_flat_pot = get_pycapsule_ptr<VectorizedFlatPotentialFunction>(py_pot,
                                                                                                  params.func_name());
                            } else {
                                if (debug_print()) printf("  > getting pointer to vectorized potential %s (pointer: %s)\n",
                                                          get_python_repr(py_pot).c_str(),
                                                          params.func_name().c_str());
                                vec_pot = get_pycapsule_ptr<VectorizedPotentialFunction>(py_pot, params.func_name());
                            }
                            break;
                        }
                        case ThreadingMode::OpenMP:
                        case ThreadingMode::TBB:
                        case ThreadingMode::SERIAL: {
                            if (params.flat_mode()) {
                                if (debug_print()) printf("  > getting pointer to flat potential %s (pointer: %s)\n",
                                                          get_python_repr(py_pot).c_str(),
                                                          params.func_name().c_str());
                                flat_pot = get_pycapsule_ptr<FlatPotentialFunction>(py_pot,params.func_name());
                            } else {

                                if (debug_print()) printf("  > getting pointer to potential %s (pointer: %s)\n",
                                                          get_python_repr(py_pot).c_str(),
                                                          params.func_name().c_str());
                                pot = get_pycapsule_ptr<PotentialFunction>(py_pot, params.func_name());
                            }
                            break;
                        }
                        default:
                            throw std::runtime_error("unknown threading mode");


                    }
                    break;
                }
                case (2) :
                    if (rynlib::common::debug_print()) printf("  > extracting FFIModule from capsule...\n");
                    module = ffi_from_capsule(py_pot);
                    break;
                default:
                    throw std::runtime_error("unknown caller API version");
            }

        }

        template<typename T>
        FFIMethod<T> PotentialApplier::get_method() {
            auto fname = params.func_name();
            return module.get_method<T>(fname);
        }

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
            auto method = get_method<double>();

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

                // insert coordinates into Parameters
                auto call_params = params.ffi_params();
                std::string coords_key = "coords";
                auto walker = coords.get_flat_walker(which);
                auto data_ptr = std::shared_ptr<void>(walker.data(), [](double*){});
                auto shp = coords.get_shape();
                FFIArgument arg (coords_key, FFIType::Double, shp);
                FFIParameter coords_param(data_ptr, arg);
                call_params.set_parameter(coords_key, coords_param);

                // insert atoms into Parameters
                std::string atoms_key = "atoms";
                auto atdata_ptr = std::shared_ptr<void>(atoms.data(), [](std::string*){});
                std::vector<size_t> at_shp = { atoms.size() };
                FFIArgument atsarg (atoms_key, FFIType::String, at_shp);
                FFIParameter ats_param(atdata_ptr, atsarg);
                call_params.set_parameter(atoms_key, ats_param);

                pot_val = method.call(call_params);
                if (debug_print) printf("  got back energy: %f\n", pot_val);

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
                case (1): {
                    return call_1(coords, which, params.retries());
                    break;
                }
                case (2): {
                    return call_2(coords, which, params.retries());
                    break;
                }
                default:
//                    printf("> ? > ? %d\n", params.api_version());
                    throw std::runtime_error("unknown caller API version");
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
                    default: {
//                        printf("> ... > ... %d\n", params.api_version());
                        throw std::runtime_error("unknown caller API version");
                    }
                }
            }
        }

        PotValsManager PotentialApplier::call_vectorized_1(
                CoordsManager &coords,
                int retries
        ) {

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
                    pots = call_vectorized_1(coords, retries - 1);
                } else {
                    printf("Error in vectorized call %s\n", e.what());
                    pots = PotValsManager(coords.num_calls(), coords.num_walkers(), params.error_val());
                }
            }

            return pots;
        }

        // New API
        PotValsManager PotentialApplier::call_vectorized_2(
                CoordsManager &coords,
                int retries
        ) {
            PotValsManager pot_vals;

            auto debug_print = params.debug();
            auto shape = coords.get_shape();
            auto atoms = coords.get_atoms();

            if (debug_print) {
                printf("  > calling vectorized potential on %ld walkers\n", coords.num_geoms());
            }

            auto method = get_method<std::vector<double> >();
            try {

                // might need a proper copy?
                auto call_params = params.ffi_params();
                std::string key = "coords";
                auto data_ptr = std::shared_ptr<void>(coords.data(), [](RawWalkerBuffer){});
                auto shp = coords.get_shape();
                FFIArgument arg (key, FFIType::Double, shp);
                FFIParameter coords_param(data_ptr, arg);
                call_params.set_parameter(key, coords_param);

                // insert atoms into Parameters
                std::string atoms_key = "atoms";
                auto atdata_ptr = std::shared_ptr<void>(atoms.data(), [](std::string*){});
                std::vector<size_t> at_shp = {atoms.size()};
                FFIArgument atsarg (atoms_key, FFIType::String, at_shp);
                FFIParameter ats_param(atdata_ptr, atsarg);
                call_params.set_parameter(atoms_key, ats_param);

                auto poop = call_params.shape("atoms");

                auto pot_vec = method.call(call_params);
                pot_vals = PotValsManager(pot_vec, coords.num_calls());

            } catch (std::exception &e) {
                if (retries > 0) {
                    pot_vals = call_vectorized_2(coords, retries - 1);
                } else {
                    printf("  > error in vectorized call %s\n", e.what());
                    pot_vals = PotValsManager(coords.num_calls(), coords.num_walkers(), params.error_val());
                }
            }

            return pot_vals;

        };

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

            if (call_parameters().debug()) {
                printf("Calling into potential '%s' for %lu steps with %lu walkers\n",
                       call_parameters().func_name().c_str(),
                       ncalls,
                       nwalkers
                );
            }
            PotValsManager pots(ncalls, nwalkers);

            switch (mode) {
                case (ThreadingMode::OpenMP) : {
                    if (call_parameters().debug()) {
                        printf(" > caller threading using %s\n", "OpenMP");
                    }
                    ThreadingHandler::_call_omp(pots, coords);
                    break;
                }
                case (ThreadingMode::TBB) : {
                    if (call_parameters().debug()) {
                        printf(" > caller threading using %s\n", "TBB");
                    }
                    ThreadingHandler::_call_tbb(pots, coords);
                    break;
                }
                case (ThreadingMode::VECTORIZED) : {
                    if (call_parameters().debug()) {
                        printf(" > caller threading using %s\n", "internal vectorization ");
                    }
                    ThreadingHandler::_call_vec(pots, coords);
                    break;
                }
                case (ThreadingMode::PYTHON) : {
                    if (call_parameters().debug()) {
                        printf(" > caller threading using %s\n", "python-side vectorization ");
                    }
                    ThreadingHandler::_call_python(pots, coords);
                    break;

                }
                case (ThreadingMode::SERIAL) : {
                    if (call_parameters().debug()) { printf(" > caller unthreaded\n"); }
                    ThreadingHandler::_call_serial(pots, coords);
                    break;
                }
                default:
                    throw std::runtime_error("Bad threading mode?");
            }

//            printf("   wwwwwoooasdasd %f\n", pots.vector()[0]);
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

//            size_t which_dat[2] = {n, i};
            std::vector<size_t> which = {n, i}; // C++17 support
            Real_t pot_val = pot_caller.call(coords, which);

            if (pot_caller.call_parameters().debug()) {
                printf("  inserting energy into array at (%lu, %lu)\n", n, i);
            }

            pots.assign(n, i, pot_val);
        }

        void ThreadingHandler::_call_vec(PotValsManager &pots, CoordsManager &coords) {
            pots = pot.call_vectorized(coords);
        }

        void ThreadingHandler::_call_python(PotValsManager &pots, CoordsManager &coords) {
            pots = pot.call_python(coords);
        }

        void ThreadingHandler::_call_serial(PotValsManager &pots, CoordsManager &coords
        ) {

            auto atoms = coords.get_atoms();
            auto ncalls = coords.num_calls();
            auto nwalkers = coords.num_walkers();

            auto total_walkers = ncalls * nwalkers;
//            auto debug_print = args.debug_print;

            for (size_t w = 0; w < total_walkers; w++) {
                _loop_inner(
                        pots,
                        coords,
                        pot,
                        nwalkers,
                        w
                );
            }

//            printf(">>>> boopy %f\n", pots.vector()[0]);
        }

        void ThreadingHandler::_call_omp(
                PotValsManager &pots,
                CoordsManager &coords
        ) {

#ifdef _OPENMP
            Py_BEGIN_ALLOW_THREADS
            auto atoms = coords.get_atoms();
            auto ncalls = coords.num_calls();
            auto nwalkers = coords.num_walkers();

            auto total_walkers = ncalls * nwalkers;
//            auto debug_print = args.debug_print;

            #pragma omp parallel for
            for (size_t w = 0; w < total_walkers; w++) {
                if (debug_print()) {
                    printf("  > calling walker %lu on thread %d", w, omp_get_thread_num());
                }
                _loop_inner(
                        pots,
                        coords,
                        pot,
                        nwalkers,
                        w
                );
            }
            Py_END_ALLOW_THREADS
#else
            throw std::runtime_error("OpenMP not installed");

#endif
        }

        void ThreadingHandler::_call_tbb(
                PotValsManager &pots,
                CoordsManager &coords
        ) {
#ifdef _TBB
            Py_BEGIN_ALLOW_THREADS
            auto atoms = coords.get_atoms();
            auto ncalls = coords.num_calls();
            auto nwalkers = coords.num_walkers();

            auto total_walkers = ncalls * nwalkers;
//            auto debug_print = args.debug_print;

            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, total_walkers),
                    [&](const tbb::blocked_range <size_t> &r) {
                        for (size_t w = r.begin(); w < r.end(); ++w) {
                            if (debug_print()) {
                                printf("  > calling walker %lu on thread %d", w, tbb::task_arena::current_thread_index());
                            }
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
            Py_END_ALLOW_THREADS
#else
            throw std::runtime_error("TBB not installed");
#endif
        }

    } // namespace PlzNumbers
}