//
// Created by Mark Boyer on 1/24/20.
//

#ifndef RYNLIB_WALKERPROPAGATOR_H

#include <random>
#include <unordered_map>
#include "RynTypes.hpp"

typedef std::normal_distribution<Real_t> RNG;
typedef std::vector<RNG> RNGVector;

typedef int CoreID;
typedef std::vector<CoreID> CoreIDs;
typedef int WalkerID;
typedef std::vector<WalkerID> WalkerIDs;

/*
 * WalkerPropagator Setup
 *
 * the new, minimal-communication setup for RynLib makes use of a WalkerPropagator class to handle the core walker
 * modifications and stuff
 *
 */

// We'll set up a WalkerPropagator class that will be initialized on a per-core basis from the passed
// array of walkers. This will also get the atom names and the sigmas at the initialization.
// We'll bind the object pointer to a PyCapsule object that we can then attach to our simulation for further calls.
// This thing will maintain a normal_distribution object that will be used to propagate the walkers forward and to
// calculate all of the potential values in turn

class WalkerPropagator {

    // I'm just gonna make this all public because... honestly why not?
public:
    bool initialized;

    // these are needed to actually compute the potential value
    Names atoms;
    Configurations walkers;

    // this is used for actually propagating the system forward
    Weights sigmas;
    int prop_steps;

    // these will be unused but potentially useful
    CoreID core_num;
    WalkerIDs walker_ids;
    CoreIDs walker_map; // will be the same for all cores but computed only once
    int world_size;

    std::default_random_engine engine;
    RNGVector distributions;

    // this is basically the only method that matters -- this is what will get generate and return all the potential values
    RawPotentialBuffer get_pots();

    // initialization function so that we can do this separately from construction
    void init(
            Names atoms,
            Configurations walkers,
            Weights sigmas,
            int prop_steps,
            int world_size
    ){
        this->atoms = atoms;
        this->walkers = walkers;
        this->sigmas = sigmas;
        this->prop_steps = prop_steps;
        this->world_size = world_size;
        distributions = RNGVector(3);
        for ( int i = 0; i < 3; i++ ) {
            distributions[i] = RNG(0, sigmas[i]);
        }

        auto num_walkers = walkers.size();
        auto total_walkers = world_size * num_walkers;

        // compute the walker_map
        CoreID core = 0;
        walker_map.reserve(total_walkers);
        for (WalkerID walker = 0; walker++; walker < total_walkers) {
            walker_map[walker] = core;
            if ( walker % num_walkers == num_walkers - 1) { // right before we roll over
                core++;
            }
        }

        initialized = true;
    }

    WalkerPropagator(
            bool initialized = false
    ){
        this->initialized = initialized;
    };
};
struct CoreSpec {
    // we use this to package up which cores have which walkers so we can easily feed this back to the
    // python side of things for management
    CoreIDs cores;
    std::vector<WalkerIDs> walkers;
};


#define RYNLIB_WALKERPROPAGATOR_H

#endif //RYNLIB_WALKERPROPAGATOR_H
