
#include "RynTypes.hpp"
#include "PyAllUp.hpp"

void _printOutWalkerStuff( Coordinates walker_coords, const std::string &bad_walkers ) {
    if (!bad_walkers.empty()) {
        const char* fout = bad_walkers.c_str();
        FILE *err = fopen(fout, "a");
        fprintf(err, "This walker was bad: ( ");
        for (size_t i = 0; i < walker_coords.size(); i++) {
            fprintf(err, "(%f, %f, %f)", walker_coords[i][0], walker_coords[i][1], walker_coords[i][2]);
            if (i < walker_coords.size() - 1) {
                fprintf(err, ", ");
            }
        }
        fprintf(err, " )\n");
        fclose(err);
    } else {
        printf("This walker was bad: ( ");
        for ( size_t i = 0; i < walker_coords.size(); i++) {
            printf("(%f, %f, %f)", walker_coords[i][0], walker_coords[i][1], walker_coords[i][2]);
            if ( i < walker_coords.size()-1 ) {
                printf(", ");
            }
        }
        printf(" )\n");
    }
}


// Basic method for computing a potential via the global potential bound in POOTY_PATOOTY
double _doopAPot(
        Coordinates &walker_coords,
        Names &atoms,
        PotentialFunction pot_func,
        std::string &bad_walkers_file,
        double err_val,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats,
        int retries = 3
        ) {
    double pot;

    try {
        pot = pot_func(walker_coords, atoms, extra_bools, extra_ints, extra_floats);
    } catch (std::exception &e) {
        if (retries > 0){
            return _doopAPot(
                    walker_coords, atoms, pot_func, bad_walkers_file, err_val,
                    extra_bools, extra_ints, extra_floats,
                    retries-1
                    );
        } else {
            PyErr_SetString(PyExc_ValueError, e.what());
            _printOutWalkerStuff(walker_coords, bad_walkers_file);
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

// This is the first of a set of methods written so as to _directly_ communicate with the potential and things
// based off of a set of current geometries and the atom names.
// We're gonna move to a system where we do barely any communication and instead ask each core to basically propagate
// its own walker(s) directly and compute energies and all that without needing to be directed to by the main core
// it'll propagate and compute on its own and only take updates from the parent when it needs to

PotentialArray _mpiGetPot(
        PyObject* manager,
        PotentialFunction pot,
        RawWalkerBuffer raw_data,
        Names &atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        PyObject* bad_walkers_file,
        double err_val,
        bool vectorized_potential,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats
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
    int world_rank = _FromInt(wr);
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
    int walkers_to_core = ncalls * num_walkers_per_core;

    RawWalkerBuffer walker_buf = (RawWalkerBuffer) malloc(walkers_to_core * walker_cnum * sizeof(Real_t));

    // Scatter data buffer to processors
    PyObject* scatter = PyObject_GetAttrString(manager, "_scatter_walkers");
    ScatterFunction scatter_walkers = (ScatterFunction) PyCapsule_GetPointer(scatter, "_Scatter_Walkers");
    scatter_walkers(
            manager,
            raw_data,  // raw data buffer to chunk up
            walkers_to_core,
            walker_cnum, // three coordinates per atom per num_atoms per walker
            walker_buf // raw array to write into
    );
    Py_XDECREF(scatter);

    // Allocate a coordinate array to pull data into
    Coordinates walker_coords (num_atoms, Point(3));

    // Do the same with the potentials
    // walkers_to_core is the number of calls * the number of walkers per core
    // We initialize a _single_ potential vector to handle all of this junk because the data is coming out of python
    // as a single vector

    // The annoying thing is that the buffer is oriented like:
    //   [
    //      walker_0(t=0), walker_0(t=1), ... walker_0(t=n),
    //      walker_1(t=0), walker_1(t=1), ... walker_1(t=n),
    //      ...,
    //      walker_m(t=0), walker_m(t=1), ... walker_m(t=n)
    //   ]
    // Each chunk looks like:
    //   [
    //      walker_i(t=0), walker_i(t=1), ... walker_i(t=n),
    //      ...,
    //      walker_(i+k)(t=0), walker_(i+k)(t=1), ... walker_(i+k)(t=n)
    //   ]

    PotentialVector pots(walkers_to_core, 0);
    PyObject* pyStr = NULL;
    std::string bad_file = _GetPyString(bad_walkers_file, pyStr);
    Py_XDECREF(pyStr);
    for (int i = 0; i < walkers_to_core; i++) {
        // Some amount of wasteful copying but ah well
        walker_coords = _getWalkerCoords(walker_buf, i, num_atoms);
        pots[i] = _doopAPot(
                walker_coords,
                atoms,
                pot,
                bad_file,
                err_val,
                extra_bools,
                extra_ints,
                extra_floats
                );
    }
    //   [
    //      pot_i(t=0), pot_i(t=1), ... pot_i(t=n),
    //      ...,
    //      pot_(i+k)(t=0), pot_(i+k)(t=1), ... pot_(i+k)(t=n)
    //   ]

    // we don't work with the walker data anymore?
    free(walker_buf);

    // receive buffer -- needs to be the number of walkers total in the system,
    // so we take the number of walkers and multiply it into the number of calls we make
    RawPotentialBuffer pot_buf = NULL;
    if ( world_rank == 0) {
        pot_buf = (RawPotentialBuffer) malloc(ncalls * num_walkers * sizeof(Real_t));
    }
    PyObject* gather = PyObject_GetAttrString(manager, "_gather_walkers");
    GatherFunction gather_walkers = (GatherFunction) PyCapsule_GetPointer(gather, "_Gather_Walkers");
    gather_walkers(
            manager,
            pots,
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
        // at this point we have (num_walkers, ncalls) shaped potVals array, too, so I'm just gonna copy it
        // I think I _also_ copy it again downstream but, to be honest, I don't care???
        for (int call = 0; call < ncalls; call++) {
            for (int walker = 0; walker < num_walkers; walker++) {
                potVals[walker][call] = pot_buf[ind2d(walker, call, num_walkers, ncalls)];
            }
        }
        free(pot_buf);
    }

    return potVals;
}

PotentialArray _noMPIGetPot(
        PotentialFunction pot,
        double* raw_data,
        Names &atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        PyObject* bad_walkers_file,
        double err_val,
        bool vectorized_potential,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats
) {
    // currently I have nothing to manage an independently vectorized potential but maybe someday I will
    PotentialArray potVals(num_walkers, PotentialVector(ncalls, 0));
    PyObject* pyStr = NULL;
    std::string bad_file = _GetPyString(bad_walkers_file, pyStr);
    Py_XDECREF(pyStr);
    Coordinates walker_coords;
    for (int n = 0; n < ncalls; n++) {
        for (int i = 0; i < num_walkers; i++) {
            walker_coords = _getWalkerCoords2(raw_data, n, i, ncalls, num_walkers, num_atoms);
            potVals[i][n] = _doopAPot(
                    walker_coords,
                    atoms,
                    pot,
                    bad_file,
                    err_val,
                    extra_bools,
                    extra_ints,
                    extra_floats
            );
        }
    }
    return potVals;

}