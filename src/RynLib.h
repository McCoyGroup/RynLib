
#include "Python.h"

#ifdef SADBOYDEBUG

// Empty do nothing debug config for cleanest debugging

#else

#include <vector>
#include <string>

#ifdef IM_A_REAL_BOY

#include "dmc_interface.h"
#include "mpi.h"
// MillerGroup_entosPotential is really in libentos but this predeclares it

#else
// for testing we roll our own which always spits out 52

double MillerGroup_entosPotential
        (const std::vector< std::vector<double> > , const std::vector<std::string>, bool hf_only = false) {

    return 52.0;

}

#endif //ENTOS_ML_DMC_INTERFACE_H

// We'll do a bunch of typedefs and includes and stuff to make it easier to work with/debug this stuff

#include <stdio.h>
#include <stdexcept>
#include <random>
#include <map>

typedef double Real_t; // easy hook in case we wanted to use a different precision object or something in the future
typedef Real_t* RawWalkerBuffer;
typedef Real_t* RawPotentialBuffer;
typedef std::vector<Real_t> Point;
typedef Point PotentialVector;
typedef Point Weights;
typedef std::vector< Point > Coordinates;
typedef Coordinates PotentialArray;
typedef std::vector< Coordinates > Configurations;
typedef std::string Name;
typedef std::vector<std::string> Names;
typedef std::normal_distribution<Real_t> RNG;
typedef std::vector<RNG> RNGVector;

typedef int CoreID;
typedef std::vector<CoreID> CoreIDs;
typedef int WalkerID;
typedef std::vector<WalkerID> WalkerIDs;
typedef std::map<CoreID, WalkerIDs> CoreWalkerMap;
typedef std::map<CoreID, CoreWalkerMap> CoreCoreMap;

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

        std::default_random_engine engine;
        RNGVector distributions;

        // this is basically the only method that matters -- this is what will get generate and return all the potential values
        RawPotentialBuffer get_pots();

        // initialization function so that we can do this separately from construction
        void init(
                Names atoms,
                Configurations walkers,
                Weights sigmas,
                int prop_steps
        ){
            this->atoms = atoms;
            this->walkers = walkers;
            this->sigmas = sigmas;
            this->prop_steps = prop_steps;
            distributions = RNGVector(3);
            for ( int i = 0; i < 3; i++ ) {
                distributions[i] = RNG(0, sigmas[i]);
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



/*
 * Python Interface
 *
 * these are the methods that will actually be exported and available to python
 * nothing else will be visible directly, so we need to make sure that this set is sufficient for out purposes
 *
 */
static PyObject *RynLib_callPot
    ( PyObject *, PyObject * );

static PyObject *RynLib_callPotVec
    ( PyObject *, PyObject * );

static PyObject *RynLib_testPot
    ( PyObject *, PyObject * );

static PyObject *RynLib_initializeMPI
    ( PyObject *, PyObject * );

static PyObject *RynLib_finalizeMPI
    ( PyObject *, PyObject * );

static PyObject *RynLib_holdMPI
        ( PyObject *, PyObject * );

#endif