//
// Created by Mark Boyer on 1/29/20.
//

#include "Python.h"
#include "RynTypes.hpp"
#include "Potators.hpp"
#include "mpi.h"

void _mpiInit(int* world_size, int* world_rank) {
    // Initialize MPI state
    int did_i_do_good_pops = 0;
    MPI_Initialized(&did_i_do_good_pops);
    if (!did_i_do_good_pops){
        MPI_Init(NULL, NULL);
    };
    MPI_Comm_size(MPI_COMM_WORLD, world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, world_rank);
    // printf("This many things %d and I am here %d\n", *world_size, *world_rank);
}

void _mpiFinalize() {
    int did_i_do_bad_pops = 0;
    MPI_Finalized(&did_i_do_bad_pops); // need to check if we called Init once already
    if (!did_i_do_bad_pops){
        MPI_Finalize();
    };
}

void _mpiBarrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}


// This is the first of a set of methods written so as to _directly_ communicate with the potential and things
// based off of a set of current geometries and the atom names.
// We're gonna move to a system where we do barely any communication and instead ask each core to basically propagate
// its own walker(s) directly and compute energies and all that without needing to be directed to by the main core
// it'll propagate and compute on its own and only take updates from the parent when it needs to

PotentialArray _mpiGetPot(
        RawWalkerBuffer raw_data,
        const Names &atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank
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

    auto walker_buf = (RawWalkerBuffer) malloc(walkers_to_core * walker_cnum * sizeof(Real_t));
    // Scatter data buffer to processors
    MPI_Scatter(
            raw_data,  // raw data buffer to chunk up
            walkers_to_core * walker_cnum, // three coordinates per atom per num_atoms per walker
            MPI_DOUBLE, // coordinates stored as doubles
            walker_buf, // raw array to write into
            walkers_to_core * walker_cnum, // three coordinates per atom per num_atoms per walker
            MPI_DOUBLE, // coordinates stored as doubles
            0, // root caller
            MPI_COMM_WORLD // communicator handle
    );

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
    for (int i = 0; i < walkers_to_core; i++) {
        // Some amount of wasteful copying but ah well
        walker_coords = _getWalkerCoords(walker_buf, i, num_atoms);
        pots[i] = _doopAPot(walker_coords, atoms);
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
    RawPotentialBuffer pot_buf;
    if ( world_rank == 0) {
        pot_buf = (RawPotentialBuffer) malloc(ncalls * num_walkers * sizeof(Real_t));
    }
    MPI_Gather(
            pots.data(),
            walkers_to_core, // number of walkers fed in
            MPI_DOUBLE, // coordinates stored as doubles
            pot_buf, // buffer to get the potential values back
            walkers_to_core, // number of walkers fed in
            MPI_DOUBLE, // coordinates stored as doubles
            0, // where they should go
            MPI_COMM_WORLD // communicator handle
    );

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
        for (size_t call = 0; call < ncalls; call++) {
            for (size_t walker = 0; walker < num_walkers; walker++) {
                potVals[walker][call] = pot_buf[ind2d(walker, call, num_walkers, ncalls)];
            }
        }
        free(pot_buf);
    }

    return potVals;
}