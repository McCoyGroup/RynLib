
#include "RynTypes.hpp"
#include "PyAllUp.hpp"
#include "Potators.hpp"
#include <csignal>
#include <sstream>
#include <iostream>
#include <thread>
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_scheduler_init.h"
#include <mutex>

#include "wchar.h"
//using namespace tbb;

std::string _appendWalkerStr(const char* base_str, const char* msg, Coordinates &walker_coords) {
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
std::string _appendWalkerStr(const char* base_str, const char* msg, FlatCoordinates &walker_coords) {
    std::string err_msg = base_str;
    err_msg += msg;
    err_msg += "(";
    for (size_t i = 0; i < walker_coords.size()/3; i++) {
        err_msg += "(";
        for (int j = 0; j < 3; j++) {
            err_msg += std::to_string(walker_coords[i*3 + j]);
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
    const char* err_string
    ) {

    std::string err_msg = _appendWalkerStr(err_string, " \n This walker was bad: ( ", walker_coords);

    if (!bad_walkers.empty()) {
        const char* fout = bad_walkers.c_str();
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
        const char* err_string
) {

    std::string err_msg = _appendWalkerStr(err_string, " \n This walker was bad: ( ", walker_coords);
    if (!bad_walkers.empty()) {
        const char* fout = bad_walkers.c_str();
        FILE *err = fopen(fout, "a");
        fprintf(err, "%s\n", err_msg.c_str());
        fclose(err);
    } else {
        printf("%s\n", err_msg.c_str());
    }

}

void _sigillHandler( int signum ) {
    printf("Illegal instruction signal (%d) received.\n", signum );
    abort();
//    exit(signum);
}
void _sigsevHandler( int signum ) {
    printf("Segfault signal (%d) received.\n", signum );
    abort();
}

double _doopAPot(
        Coordinates &walker_coords,
        Names &atoms,
        PotentialFunction pot_func,
        std::string &bad_walkers_file,
        double err_val,
        bool debug_print,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats,
        int retries
        ) {
    double pot;


    try {
        signal(SIGSEGV, _sigsevHandler);
        signal(SIGILL, _sigillHandler);
        if (debug_print) {
            std::string walker_string = _appendWalkerStr("Walker before call: ", "", walker_coords);
            printf("%s\n", walker_string.c_str());
        }
        pot = pot_func(walker_coords, atoms, extra_bools, extra_ints, extra_floats);
        if (debug_print) {
            printf("  got back energy: %f\n", pot);
        }

    } catch (std::exception &e) {
        if (retries > 0){
            return _doopAPot(
                    walker_coords, atoms, pot_func,
                    bad_walkers_file, err_val, debug_print,
                    extra_bools, extra_ints, extra_floats,
                    retries-1
                    );
        } else {
            // pushed error reporting into bad_walkers_file
            // should probably revise yet again to print all of this stuff to python's stderr...
            if (debug_print) {
                std::string no_str="";
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
        std::string &bad_walkers_file,
        double err_val,
        bool debug_print,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats,
        int retries
) {
    double pot;


    try {

        signal(SIGSEGV, _sigsevHandler);
        signal(SIGILL, _sigillHandler);
        if (debug_print) {
            std::string walker_string = _appendWalkerStr("Walker before call: ", "", walker_coords);
            printf("%s\n", walker_string.c_str());
        }
        pot = pot_func(walker_coords, atoms, extra_bools, extra_ints, extra_floats);

    } catch (std::exception &e) {
        if (retries > 0){
            return _doopAPot(
                    walker_coords, atoms, pot_func,
                    bad_walkers_file, err_val, debug_print,
                    extra_bools, extra_ints, extra_floats,
                    retries-1
            );
        } else {
            if (debug_print) {
                std::string no_str="";
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


inline int ind2d(int i, int j, int n, int m) {
    return m * i + j;
}

// here I ignore `n` because... well I originally wrote it like that
inline int int3d(int i, int j, int k, int m, int l) {
    return (m*l) * i + (l*j) + k;
}

Coordinates _getWalkerCoords(const double* raw_data, int i, Py_ssize_t num_atoms) {
    Coordinates walker_coords(num_atoms, Point(3));
    for (int j = 0; j<num_atoms; j++) {
        for (int k = 0; k<3; k++) {
            walker_coords[j][k] = raw_data[int3d(i, j, k, num_atoms, 3)];
        }
    };
    return walker_coords;
}
FlatCoordinates _getWalkerFlatCoords(const double* raw_data, int i, Py_ssize_t num_atoms) {
    FlatCoordinates walker_coords (num_atoms*3);
    for (int j = 0; j<num_atoms; j++) {
        for (int k = 0; k<3; k++) {
            double crd = raw_data[int3d(i, j, k, num_atoms, 3)];
//            printf("...%f\n", crd);
            walker_coords[ind2d(j, k, num_atoms, 3)] = crd;
        }
    };
    return walker_coords;
}

inline int int4d(int i, int j, int k, int a, int n, int m, int l, int o) {
    return (m*l*o) * i + (l*o*j) + o*k + a;
}

// pulls data for the ith walker in the nth call
// since we start out with data that looks like (ncalls, nwalkers, ...)
Coordinates _getWalkerCoords2(const double* raw_data, int n, int i, int ncalls, int num_walkers, Py_ssize_t num_atoms) {
    Coordinates walker_coords(num_atoms, Point(3));
    for (int j = 0; j<num_atoms; j++) {
        for (int k = 0; k<3; k++) {
            walker_coords[j][k] = raw_data[int4d(n, i, j, k, ncalls, num_walkers, num_atoms, 3)];
        }
    };
    return walker_coords;
}
FlatCoordinates _getWalkerFlatCoords2(const double* raw_data, int n, int i, int ncalls, int num_walkers, Py_ssize_t num_atoms) {
    FlatCoordinates walker_coords(num_atoms*3);
    for (int j = 0; j<num_atoms; j++) {
        for (int k = 0; k<3; k++) {
            walker_coords[ind2d(j, k, num_atoms, 3)] = raw_data[int4d(n, i, j, k, ncalls, num_walkers, num_atoms, 3)];
        }
    };
    return walker_coords;
}

// This is the first of a set of methods written so as to _directly_ communicate with the potential and things
// based off of a set of current geometries and the atom names.
// We're gonna move to a system where we do barely any communication and instead ask each core to basically propagate
// its own walker(s) directly and compute energies and all that without needing to be directed to by the main core
// it'll propagate and compute on its own and only take updates from the parent when it needs to

RawWalkerBuffer _scatterWalkers(
        PyObject* manager,
        RawWalkerBuffer raw_data,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int& world_rank,
        int& walkers_to_core
        ) {
    //
    // The way this works is that we start with an array of data that looks like (ncalls, num_walkers, *walker_shape)
    // Then we have m cores such that num_walkers_per_core = num_walkers / m
    //
    // We pass these in to MPI and allow them to get distributed out as blocks of ncalls * num_walkers_per_core walkers
    // to a given core, which calculates the potential over all of them and then returns that
    //
    // At the end we have a potential array that is m * (ncalls * num_walkers_per_core) walkers and we need to make this
    // back into the clean (ncalls, num_walkers) array we expect in the end
    PyObject* ws = PyObject_GetAttrString(manager, "world_size");
    int world_size = _FromInt(ws);
    Py_XDECREF(ws);
    PyObject* wr = PyObject_GetAttrString(manager, "world_rank");
    world_rank = _FromInt(wr);
    Py_XDECREF(wr);

    // we're gonna assume the former is divisible by the latter on world_rank == 0
    // and that it's just plain `num_walkers` on every other world_rank
    int num_walkers_per_core = (num_walkers / world_size);
    if (world_rank > 0) {
        // means we're only feeding in num_walkers because we're not on world_rank == 0
        num_walkers_per_core = num_walkers;
    }

    // create a buffer for the walkers to be fed into MPI
    int walker_cnum = num_atoms*3;
    walkers_to_core = ncalls * num_walkers_per_core;

    RawWalkerBuffer walker_buf = (RawWalkerBuffer) malloc(walkers_to_core * walker_cnum * sizeof(Real_t));

    // Scatter data buffer to processors
    PyObject* scatter = PyObject_GetAttrString(manager, "scatter");
    ScatterFunction scatter_walkers = (ScatterFunction) PyCapsule_GetPointer(scatter, "Dumpi._SCATTER_WALKERS");
    scatter_walkers(
            manager,
            raw_data,  // raw data buffer to chunk up
            walkers_to_core,
            walker_cnum, // three coordinates per atom per num_atoms per walker
            walker_buf // raw array to write into
    );
    Py_XDECREF(scatter);

    return walker_buf;
}

PotentialArray _gatherPotentials(
        PyObject* manager,
        int world_rank,
        int ncalls,
        RawPotentialBuffer pot_data,
        Py_ssize_t num_walkers,
        int& walkers_to_core
        ) {
    // receive buffer -- needs to be the number of walkers total in the system,
    // so we take the number of walkers and multiply it into the number of calls we make
    RawPotentialBuffer pot_buf = NULL;
    if ( world_rank == 0) {
        pot_buf = (RawPotentialBuffer) malloc(ncalls * num_walkers * sizeof(Real_t));
    }
    PyObject* gather = PyObject_GetAttrString(manager, "gather");
    GatherFunction gather_walkers = (GatherFunction) PyCapsule_GetPointer(gather, "Dumpi._GATHER_POTENTIALS");
    gather_walkers(
            manager,
            pot_data,
            walkers_to_core, // number of walkers fed in
            pot_buf // buffer to get the potential values back
    );
    Py_XDECREF(gather);

    // convert double* to std::vector<double>
    // We currently have:
    //   [
    //      pot_0(t=0), walker_0(t=1), ... walker_0(t=n),
    //      pot_1(t=0), walker_1(t=1), ... walker_1(t=n),
    //      ...,
    //      pot_m(t=0), walker_m(t=1), ... walker_m(t=n)
    //   ]
    // And so we'll just directly copy it in?
    PotentialArray potVals(num_walkers, PotentialVector(ncalls, 0));
    if( world_rank == 0 ) {
        // At this point we have (num_walkers, ncalls) shaped potVals array, too, so I'm just gonna copy it
        //    in the dumbest, least efficient way possible (TODO: make this less dumb)
        // I _also_ copy it again downstream but, to be honest, I don't care???
        for (int call = 0; call < ncalls; call++) {
            for (int walker = 0; walker < num_walkers; walker++) {
                potVals[walker][call] = pot_buf[ind2d(walker, call, num_walkers, ncalls)];
            }
        }
        free(pot_buf);
    }

    return potVals;
}

Real_t _getPotFlat(
        RawWalkerBuffer walker_buf,
        int n, int ncalls, int i, int num_walkers,
        Py_ssize_t num_atoms,
        Names &atoms,
        FlatPotentialFunction pot,
        std::string bad_file,
        double err_val,
        bool debug_print,
        int retries,
        ExtraBools& extra_bools,
        ExtraInts& extra_ints,
        ExtraFloats& extra_floats
        ) {

    FlatCoordinates walker_coords;
    if (ncalls == -1) {
        walker_coords = _getWalkerFlatCoords(walker_buf, i, num_atoms);
    } else {
        walker_coords = _getWalkerFlatCoords2(walker_buf, n, i, ncalls, num_walkers, num_atoms);
    }

    return _doopAPot(
            walker_coords,
            atoms,
            pot,
            bad_file,
            err_val,
            debug_print,
            extra_bools,
            extra_ints,
            extra_floats,
            retries
    );
}
Real_t _getPot(
        RawWalkerBuffer walker_buf,
        int n, int ncalls, int i, int num_walkers,
        Py_ssize_t num_atoms,
        Names& atoms,
        PotentialFunction pot,
        std::string bad_file,
        double err_val,
        bool debug_print,
        int retries,
        ExtraBools& extra_bools,
        ExtraInts& extra_ints,
        ExtraFloats& extra_floats
) {
    Coordinates walker_coords;
    if (ncalls == -1) {
        walker_coords = _getWalkerCoords(walker_buf, i, num_atoms);
    } else {
        walker_coords = _getWalkerCoords2(walker_buf, n, i, ncalls, num_walkers, num_atoms);
    }
    return _doopAPot(
            walker_coords,
            atoms,
            pot,
            bad_file,
            err_val,
            debug_print,
            extra_bools,
            extra_ints,
            extra_floats,
            retries
    );
}

//std::mutex WTFLock;
class PotentialCaller {
    RawWalkerBuffer walker_buf;
    int ncalls;
    int walkers_to_core;
    Names &atoms;
    int num_atoms;
    PotentialFunction pot;
    FlatPotentialFunction flat_pot;
    ExtraBools& extra_bools;
    ExtraInts& extra_ints;
    ExtraFloats& extra_floats;
    Real_t err_val;
    bool debug_print;
    int retries;
    std::string bad_file;
    bool use_openMP;
    bool use_TBB;
    bool flat_mode;
    int ncalls_loop;
    PotentialArray pots;
    int _n_current;
    RawPotentialBuffer cur_data;

    public:

    PotentialCaller(
            RawWalkerBuffer arg_walker_buf,
            int arg_ncalls,
            int arg_walkers_to_core,
            Names &arg_atoms,
            int arg_num_atoms,
            PotentialFunction arg_pot,
            FlatPotentialFunction arg_flat_pot,
            ExtraBools& arg_extra_bools,
            ExtraInts& arg_extra_ints,
            ExtraFloats& arg_extra_floats,
            Real_t arg_err_val,
            bool arg_debug_print,
            int arg_retries,
            PyObject* bad_walkers_file,
            bool arg_use_openMP,
            bool arg_use_TBB
            ) :
            walker_buf(arg_walker_buf),
            ncalls(arg_ncalls),
            walkers_to_core(arg_walkers_to_core),
            atoms(arg_atoms),
            num_atoms(arg_num_atoms),
            pot(arg_pot),
            flat_pot(arg_flat_pot),
            extra_bools(arg_extra_bools),
            extra_ints(arg_extra_ints),
            extra_floats(arg_extra_floats),
            err_val(arg_err_val),
            debug_print(arg_debug_print),
            retries(arg_retries),
            use_openMP(arg_use_openMP),
            use_TBB(arg_use_TBB)
    {
        PyObject *pyStr = NULL;
        bad_file = _GetPyString(bad_walkers_file, pyStr);
        Py_XDECREF(pyStr);
        if (pot == NULL) {
            flat_mode = true;
        } else {
            flat_mode = false;
        }
        if (ncalls > 0) {
            ncalls_loop = ncalls;
        } else {
            ncalls_loop = 1;
        }
        pots = PotentialArray(ncalls_loop, PotentialVector(walkers_to_core, 0));
        cur_data = NULL;
        _n_current = -1;
    }

    Real_t eval_pot(int n, int i) const {
        Real_t pot_val;
        if (flat_mode) {
            pot_val = _getPotFlat(
                    walker_buf,
                    n, ncalls, i, walkers_to_core,
                    num_atoms, atoms,
                    flat_pot,
                    bad_file, err_val, debug_print, retries,
                    extra_bools, extra_ints, extra_floats
            );
        } else {
            pot_val = _getPot(
                    walker_buf,
                    n, ncalls, i, walkers_to_core,
                    num_atoms, atoms,
                    pot,
                    bad_file, err_val, debug_print, retries,
                    extra_bools, extra_ints, extra_floats
            );
        }
        return pot_val;
    }
    Real_t eval_pot(int i) const {
        return eval_pot(_n_current, i);
    }

    // Serial example
    void serial_call() {
        for (int n=0; n < ncalls_loop; n++) {
            for (int i = 0; i < walkers_to_core; i++) {
                // Some amount of wasteful copying but ah well
                Real_t pot_val = eval_pot(n, i);
                pots[n][i] = pot_val;
            }
        }
    }

    void omp_call() {

        for (int n = 0; n < ncalls_loop; n++) {
            if (debug_print) printf("OpenMP: calling over block %d of size %d\n", n, walkers_to_core);
            RawPotentialBuffer current_data;
            current_data = pots[n].data();
            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < walkers_to_core; i++) {
//                    printf("Running step (%d, %d) on thread %d...\n", n, i, omp_get_thread_num());
                    Real_t pot_val = eval_pot(n, i);
                    // We're using this in a shared memory model and I haven't yet figured out if I want to try to make this
                    // thread private or anything like that...
                    current_data[i] = pot_val;
                }
            }
        }
    }

    class TBBCaller {
        PotentialCaller *caller;
        RawPotentialBuffer data;
        size_t block_n;
        bool debug_print;

    public:
        TBBCaller(
          PotentialCaller *arg_caller,
          RawPotentialBuffer arg_data,
          size_t arg_block_n,
          bool arg_debug_print
          ) : caller(arg_caller),
            data(arg_data),
            block_n(arg_block_n),
            debug_print(arg_debug_print) {}

        void operator()(const tbb::blocked_range <size_t> &r) const {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                int this_thread = tbb::task_arena::current_thread_index();
                if (debug_print) printf("Calling %ld on thread %d!\n", i, this_thread);
                Real_t pot_val = caller->eval_pot(block_n, i);
                data[i] = pot_val;
            }
        }

        /*

        [&](const blocked_range<int>& r)
        {
        for (size_t i=r.begin(); i<r.end(); ++i)
        {
        //Calculate energy of box pair i
        nrgs[i] = energy_of_box_pair(i);
        }
        }
        */
    };
    void assign_current(int i, Real_t pot_val) {
        cur_data[i] = pot_val;
    }

    void tbb_call() {

        for (int n = 0; n < ncalls_loop; n++) {
//            int num_threads = tbb_thread_counter.get_concurrency();
            if (debug_print) printf("TBB: calling block %d of size %d\n", n, walkers_to_core);
            cur_data = pots[n].data();
            _n_current = n;
//            tbb::parallel_for(tbb::blocked_range<size_t>(0, walkers_to_core), TBBCaller(this, cur_data, n, debug_print));

            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, walkers_to_core),
                    [&](const tbb::blocked_range <size_t> &r) {
                        for (size_t i = r.begin(); i < r.end(); ++i) {
                            int this_thread = tbb::task_arena::current_thread_index();
                            if (debug_print) printf("Calling %ld on thread %d!\n", i, this_thread);
                            Real_t pot_val = eval_pot(n, i);
                            if (debug_print) printf("done with %ld on thread %d!\n", i, this_thread);
                            cur_data[i] = pot_val;
                        }
                    }
            );
        }
    }

    PotentialArray apply() {

        const auto processor_count = std::thread::hardware_concurrency();
        if (use_openMP) {
//            tbb::task_scheduler_init init(1); // try to turn off TBB?
            if (debug_print) printf("Parallelization over %d threads: %s\n", processor_count, "OpenMP");
            omp_call();
        } else if (use_TBB) {
            tbb::task_scheduler_init init(processor_count - 1); // not sure _why_ I need to do this, but the default was being set to 1?
            const auto tbb_default = tbb::task_scheduler_init::default_num_threads();
            if (debug_print) printf("Parallelization over %d threads: %s (%d by default)\n", processor_count, "TBB", tbb_default);
            tbb_call();
        } else {
            if (debug_print) printf("Serial Evaluation (%d threads):\n", processor_count);
            serial_call();
        }
        return pots;
    }

};

// The two flavors have been condensed into one, where if `pot == NULL` then we assume we're using a flat potential
PotentialArray _mpiGetPot(
        PyObject* manager,
        PotentialFunction pot,
        FlatPotentialFunction flat_pot,
        RawWalkerBuffer raw_data,
        Names &atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        PyObject* bad_walkers_file,
        double err_val,
        bool debug_print,
        int retries,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats,
        bool use_openMP,
        bool use_TBB
) {
    int walkers_to_core, world_rank;
    RawWalkerBuffer walker_buf = _scatterWalkers(manager, raw_data, ncalls, num_walkers, num_atoms, world_rank, walkers_to_core);

    PotentialCaller caller(
            walker_buf,
            -1, // we use ncalls = -1 to indicate that we've flattened everything out
            walkers_to_core,
            atoms,
            num_atoms,
            pot,
            flat_pot,
            extra_bools,
            extra_ints,
            extra_floats,
            err_val,
            debug_print,
            retries,
            bad_walkers_file,
            use_openMP,
            use_TBB
    );

    PotentialArray pots = caller.apply();

    free(walker_buf);
    PotentialArray potVals = _gatherPotentials(
            manager,
            world_rank,
            ncalls,
            pots[0].data(),
            num_walkers,
            walkers_to_core
    );
    return potVals;
}

PotentialArray _noMPIGetPot(
        PotentialFunction pot,
        FlatPotentialFunction flat_pot,
        double* raw_data,
        Names &atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        PyObject* bad_walkers_file,
        double err_val,
        bool debug_print,
        int retries,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats,
        bool use_openMP,
        bool use_TBB
) {
    // currently I have nothing to manage an independently vectorized potential but maybe someday I will
    PotentialCaller caller(
            raw_data,
            ncalls,
            num_walkers,
            atoms,
            num_atoms,
            pot,
            flat_pot,
            extra_bools,
            extra_ints,
            extra_floats,
            err_val,
            debug_print,
            retries,
            bad_walkers_file,
            use_openMP,
            use_TBB
    );

    PotentialArray pots = caller.apply();

    return pots;

}

PyObject* _mpiGetPyPot(
        PyObject* manager,
        PyObject* pot_func,
        RawWalkerBuffer raw_data,
        PyObject* atoms,
        PyObject* extra,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms
) {

    int walkers_to_core, world_rank;
    RawWalkerBuffer walker_buf = _scatterWalkers(manager, raw_data, ncalls, num_walkers, num_atoms, world_rank, walkers_to_core);

    // We can just take the buffer and directly turn it into a NumPy array
    PyObject* walkers = _fillWalkersNumPyArray(walker_buf, walkers_to_core, num_atoms);

    if (walkers == NULL) {
        free(walker_buf);
        return NULL;
    }

    PyObject* args = PyTuple_Pack(3, walkers, atoms, extra);
    // We use SET_ITEM not SetItem because we _don't_ want to give our references to `args`
//    PyTuple_SET_ITEM(args, 0, walkers);
//    PyTuple_SET_ITEM(args, 1, atoms);
//    PyTuple_SET_ITEM(args, 2, extra);

    PyObject* pot_vals = PyObject_CallObject(pot_func, args);
    if (pot_vals == NULL) {
        Py_XDECREF(args);
        Py_XDECREF(walkers);
        return NULL;
    }

    RawPotentialBuffer pots = _GetDoubleDataArray(pot_vals);

    Py_XDECREF(args);

    // we don't work with the walker data at this point
    free(walker_buf);
//    Py_XDECREF(walkers); For some reason with this on we get a segfault? Need to track down why, unless it's an issue with PyTuple_SET_ITEM

    RawPotentialBuffer pot_buf = NULL;
    PyObject *potVals = NULL;
    if ( world_rank == 0) {
        potVals = _getNumPyArray(num_walkers, ncalls, "float");
        if (potVals == NULL) return NULL;
        pot_buf = _GetDoubleDataArray(potVals);
    }
    PyObject* gather = PyObject_GetAttrString(manager, "gather");
    GatherFunction gather_walkers = (GatherFunction) PyCapsule_GetPointer(gather, "Dumpi._GATHER_POTENTIALS");
    if (gather_walkers == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Couldn't get gather pointer");
        return NULL;
    }

    gather_walkers(
            manager,
            pots,
            walkers_to_core, // number of walkers fed in
            pot_buf // buffer to get the potential values back
    );
    Py_XDECREF(gather);
    Py_XDECREF(pot_vals);


    if ( world_rank > 0 ) {
        Py_RETURN_NONE;
    } else {
       return potVals;
    }

}
