#include "RynLib.h"

#ifdef SADBOYDEBUG

static PyMethodDef RynLibMethods[] = {
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initRynLib(void)
{
    (void) Py_InitModule("RynLib", RynLibMethods);
}

#else

#if PY_MAJOR_VERSION == 3
const char *_GetPyString( PyObject* s, const char *enc, const char *err, PyObject *pyStr) {
    pyStr = PyUnicode_AsEncodedString(s, enc, err);
    if (pyStr == NULL) return NULL;
    const char *strExcType =  PyBytes_AsString(pyStr);
//    Py_XDECREF(pyStr);
    return strExcType;
}
const char *_GetPyString( PyObject* s, PyObject *pyStr) {
    // unfortunately we need to pass the second pyStr so we can XDECREF it later
    return _GetPyString( s, "utf-8", "strict", pyStr); // utf-8 is safe since it contains ASCII fully
    }

Py_ssize_t _FromInt( PyObject* int_obj ) {
    return PyLong_AsSsize_t(int_obj);
}
#else
const char *_GetPyString( PyObject* s ) {
    return PyString_AsString(s);
}
const char *_GetPyString( PyObject* s, PyObject *pyStr ) {
    // just to unify the 2/3 interface
    return _GetPyString( s );
    }
Py_ssize_t _FromInt( PyObject* int_obj ) {
    return PyInt_AsSsize_t(int_obj);
}
#endif

Names _getAtomTypes( PyObject* atoms, Py_ssize_t num_atoms ) {

    Names mattsAtoms(num_atoms);
    for (int i = 0; i<num_atoms; i++) {
        PyObject* atom = PyList_GetItem(atoms, i);
        PyObject* pyStr = NULL;
        const char* atomStr = _GetPyString(atom, pyStr);
        Name atomString = atomStr;
        mattsAtoms[i] = atomString;
//        Py_XDECREF(atom);
        Py_XDECREF(pyStr);
    }

    return mattsAtoms;
}

Py_buffer _GetDataBuffer(PyObject *data) {
    Py_buffer view;
    PyObject_GetBuffer(data, &view, PyBUF_CONTIG_RO);
    return view;
}

double *_GetDoubleDataBufferArray(Py_buffer *view) {
    double *c_data;
    if ( view == NULL ) return NULL;
    c_data = (double *) view->buf;
    if (c_data == NULL) {
        PyBuffer_Release(view);
    }
    return c_data;
}

double *_GetDoubleDataArray(PyObject *data) {
    Py_buffer view = _GetDataBuffer(data);
    double *array = _GetDoubleDataBufferArray(&view);
//    CHECKNULL(array);
    return array;
}

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

#ifdef IM_A_REAL_BOY

void _printOutWalkerStuff( Coordinates walker_coords ) {
    FILE * err = fopen("bad_walkers.txt", "a");
    fprintf(err, "This walker was bad: ( ");
    for ( size_t i = 0; i < walker_coords.size(); i++) {
        fprintf(err, "(%f, %f, %f)", walker_coords[i][0], walker_coords[i][1], walker_coords[i][2]);
        if ( i < walker_coords.size()-1 ) {
            fprintf(err, ", ");
        }
    }
    fprintf(err, " )\n");
    fclose(err);
}
#else
void _printOutWalkerStuff( Coordinates walker_coords ) {
    printf("This walker was bad: ( ");
    for ( size_t i = 0; i < walker_coords.size(); i++) {
        printf("(%f, %f, %f)", walker_coords[i][0], walker_coords[i][1], walker_coords[i][2]);
        if ( i < walker_coords.size()-1 ) {
            printf(", ");
        }
    }
    printf(" )\n");
}
#endif

// Basic method for computing a potential via MillerGroup_entosPotential
double _doopAPot(const Coordinates &walker_coords, const Names &atoms) {
    double pot;

    try {
        // pot = MillerGroup_entosPotential(walker_coords, atoms, true); // use only hf
        pot = MillerGroup_entosPotential(walker_coords, atoms); // use ml as well
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
//        _printOutWalkerStuff(walker_coords);
        pot = 1.0e9;
    }

    return pot;
};


PyObject *_getNumPyZerosMethod() {
    PyObject *array_module = PyImport_ImportModule("numpy");
    if (array_module == NULL) return NULL;
    PyObject *builder = PyObject_GetAttrString(array_module, "zeros");
    Py_XDECREF(array_module);
    if (builder == NULL) return NULL;
    return builder;
};

PyObject *_getNumPyArray(
        int n,
        int m,
        const char *dtype
        ) {
    // Initialize NumPy array of correct size and dtype
    PyObject *builder = _getNumPyZerosMethod();
    if (builder == NULL) return NULL;
    PyObject *dims = Py_BuildValue("((ii))", n, m);
    Py_XDECREF(builder);
    if (dims == NULL) return NULL;
    PyObject *kw = Py_BuildValue("{s:s}", "dtype", dtype);
    if (kw == NULL) return NULL;
    PyObject *pot = PyObject_Call(builder, dims, kw);
    Py_XDECREF(kw);
    Py_XDECREF(dims);
    return pot;
}

// NumPy Communication Methods
PyObject *_fillNumPyArray(
        const PotentialArray &pot_vals,
        const int ncalls,
        const int num_walkers
        ) {

    // Initialize NumPy array of correct size and dtype
    PyObject *pot = _getNumPyArray(ncalls, num_walkers, "float");
    if (pot == NULL) return NULL;
    double *data = _GetDoubleDataArray(pot);
    for (int i = 0; i < ncalls; i++) {
        memcpy(
                // where in the data array memory to start copying to
                data + num_walkers * i,
                // where in the potential array to start copying from
                pot_vals[i].data(),
                // what
                sizeof(double) * num_walkers
                );
    };
    return pot;
}

#ifdef IM_A_REAL_BOY

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
    PotentialVector pots(walkers_to_core, 0);
    for (int i = 0; i < walkers_to_core; i++) {
        // Some amount of wasteful copying but ah well
        walker_coords = _getWalkerCoords(walker_buf, i, num_atoms);
        pots[i] = _doopAPot(walker_coords, atoms);
    }

    // we don't work with the walker data anymore?
    free(walker_buf);

    RawPotentialBuffer pot_buf;
    if ( world_rank == 0) {
        pot_buf = (RawPotentialBuffer) malloc(ncalls * num_walkers * sizeof(Real_t));
    }
    // receive buffer -- needs to be the number of walkers total in the system
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
    PotentialArray potVals(ncalls, PotentialVector(num_walkers, 0));
    if( world_rank == 0 ) {
        // at this point we have chunks where we have world_size number of blocks of length ncalls * num_walkers_per_core
        // we _want_ the data to come out as an (ncalls, num_walkers)
        for (size_t call = 0; call < ncalls; call++) {
            for (size_t walker = 0; walker < num_walkers; walker++) {
                potVals[call][walker] = pot_buf[ind2d(call, walker, ncalls, num_walkers)];
            }
        }
    }
    free(pot_buf);

    return potVals;
}


/*
 * Targeted, asynchronous MPI methods for walker broadcasting and whatnot
 *
 * We'll make use of a core-specific CORE_PROPAGATOR which will keep track of which walkers are on that given node
 *
 * On each loop we'll ask for the next n potential values, do the branching on the python side, send out a flag to all
 * the nodes telling them whether or not they need to (a) send their walker (b) update their walker or (c) do nothing
 * the ones that need to send will send at then these will be sent out to the ones that need to recieve
 *
 */


/*

 // Temporarily turning off all this

WalkerPropagator CORE_PROPAGATOR; // This is the specific WalkerPropagator we're gonna be using for a given core
void _mpiSetUpSimulation(
        RawWalkerBuffer raw_data,
        const Names &atoms,
        const Weights &sigmas,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank
        ) {

    int num_walkers_per_core = (num_walkers / world_size); // we're gonna assume the former is divisible by the latter

    // create a buffer for the walkers to be fed into MPI
    int walker_size = num_atoms*3*sizeof(Real_t);
    auto walker_buf = (RawWalkerBuffer) malloc(num_walkers_per_core*walker_size);

    // Scatter data buffer to processors
    MPI_Scatter(
            raw_data,  // raw data buffer to chunk up
            num_walkers_per_core*walker_size, // three coordinates per atom per num_atoms per walker
            MPI_DOUBLE, // coordinates stored as doubles
            walker_buf, // raw array to write into
            num_walkers_per_core*walker_size, // single energy
            MPI_DOUBLE, // energy returned as doubles
            0, // root caller
            MPI_COMM_WORLD // communicator handle
    );

    // I might be using too much memory here? But ah well
    Configurations walkers (num_walkers_per_core, Coordinates(num_atoms, Point(3)));
    for (int n = 0; n < num_walkers_per_core; n++){
        walkers[n] = _getWalkerCoords(walker_buf, n, num_atoms);
    }
    free(walker_buf);

    // On each core parse these back out, allocate them into a Configurations array, init the CORE_PROPAGATOR
    CORE_PROPAGATOR.init(
            atoms,
            walkers,
            sigmas,
            ncalls
            );

    // https://stackoverflow.com/a/48168458/5720002
    // If MPI_Scatter couldn't promise us a specific type of ordering we'd need to do some work here to get the
    // appropriate ordering out. Instead we can just make use of that fact to compute the initial ordering once on the
    // python size

}


// There's a hard-limit implict here that we won't try to run a job with more than 10000 walkers / core
// I chose to do this as a struct instead of an enum because I felt that it was nicer to read COMM_FLAGS.___
struct COMM_FLAGS_t {
    int DONT_SEND_WALKER;
    int SEND_WALKER;
    int UPDATE_WALKER;
} COMM_FLAGS = {0, 10000, 20000};

struct DecodedMessage {
    int what_do;
    int how_many;
};

DecodedMessage decode_send_message(const int msg){
    // we're gonna assume now that we've gotten out a number from our message that is greater than 10000
    // we now need to piece back how many walkers we need to work with and we're gonna then encode the position info
    // in the raw walker buffer
    DecodedMessage decode;
    if (msg > COMM_FLAGS.UPDATE_WALKER) {
        decode.what_do = COMM_FLAGS.UPDATE_WALKER;
        decode.how_many = msg % decode.what_do;
    } else if (msg > COMM_FLAGS.SEND_WALKER) {
        decode.what_do = COMM_FLAGS.SEND_WALKER;
        decode.how_many = msg % decode.what_do;
    } else {
        decode.what_do = COMM_FLAGS.DONT_SEND_WALKER;
        decode.how_many = 0;
    }

    return decode;
};

RawWalkerBuffer _mpiGetWalkersFromCores(
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank
) {

    // We'll use this as a way to get _all_ of the walker data out of the simulation
    // it won't be used much, I imagine, but as long as it gets called on all of the cores it should work...
    // maybe every like 100th timestep or something we'll get back all of our walker data, you know?

    int walker_size = num_atoms*3*sizeof(Real_t);
    int num_walkers_per_core = num_walkers / world_size;
    if (world_rank == 0) {
        // allocate buffer for getting the walkers back out of the simulation
        auto walker_buf = (RawWalkerBuffer) malloc(num_walkers*walker_size);
    } else {
        RawWalkerBuffer walker_buf;
    }

    MPI_Gather(
            CORE_PROPAGATOR.data(),
            num_walkers_per_core*walker_size, // number of walkers fed in
            MPI_DOUBLE,
            walker_buf, // buffer to get the potential values back
            num_walkers_per_core*walker_size, // number of walkers fed in
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
    );

    return walker_buf;

}

RawPotentialBuffer _mpiGetPotFromCores(
        Py_ssize_t num_walkers,
        int world_size,
        int world_rank
) {
    // this'll be called by every core so that the Gather actually works

    int num_walkers_per_core = num_walkers / world_size;
    if (world_rank == 0) {
        // allocate buffer for getting the potential values back out of the simulation
        auto pot_buf = (RawPotentialBuffer) malloc(num_walkers);
    } else {
        RawPotentialBuffer pot_buf = NULL;
    }

    MPI_Gather(
            CORE_PROPAGATOR.get_pots(),
            CORE_PROPAGATOR.prop_steps * num_walkers_per_core, // number of steps calculated
            MPI_DOUBLE,
            pot_buf, // buffer to get the potential values back
            CORE_PROPAGATOR.prop_steps * num_walkers_per_core, // number of walkers fed in
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
    );

    return pot_buf;

}

Configurations _mpiCommunicateWalkersToCores(
        CoreWalkerMap birthed,
        CoreWalkerMap dead,
        int world_size,
        int world_rank
        ) {

    // This would be the dream:
    //   we tell a _specific_ core that we want its current walker and it feeds it back to us
    //   the issue here is that the core has no asynchronous listener in place to know to listen for that message
    //   and even if it did that would slow down the HF part of the calculation unless I could put some kind of "pause"
    //   in place -- this would probably be manageable, but likely isn't worth it
    //
    // Instead we're gonna do this like so:
    //   we _asynchronously_ broadcast a flag to all cores telling them to give us their walkers or not
    //   at the same time we set up the right of listeners on rank_0 for those cores which we asked for data
    //   then after doing that we'll of course have to use MPI_Waitall on the specific recvs request set, at which point
    //   we can copy out the walker data into a Configurations array

    // broadcast to the cores that don't need to do anything that they're good to continue
    CoreMap::const_iterator core_spec;
    CoreWalkerMap::const_iterator core_spec;
    MPI_Request *req;
    for (CoreID core = 0; core < world_size; core++ ) {
        core_spec = birthed.find(core);
        if (core_spec == birthed.end()) {
            core_spec = dead.find(core);
            if (core_spec == dead.end()) {
                MPI_Isend(
                        &COMM_FLAGS.DONT_SEND_WALKER,
                        1,
                        MPI_INTEGER,
                        core,
                        0, // not sure what this tag is...
                        MPI_COMM_WORLD,
                        req
                        );
            }
        }
    }

    // iterate over the birthed walkers -> set up a Recv for each and just get all of the walkers out of a core?
    //  --> probably best for now
    std::vector<MPI_Request> recvs(birthed.size());
    int i = 0;
    for (auto it birthed.begin(); it != birthed.end(); ++it) {
        MPI_Irecv(
                &COMM_FLAGS.DONT_SEND_WALKER,
                1,
                MPI_INTEGER,
                core,
                0, // not sure what this tag is...
                MPI_COMM_WORLD,
                req
        );


    }


}


 */

#else

void _mpiInit(int* world_size, int* world_rank) {
    *world_size = 1;
    *world_rank = 0;
}

void _mpiFinalize() {
    // boop
}

void _mpiBarrier() {
    // boop
}

PotentialArray _mpiGetPot(
        double* raw_data,
        Names atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank
        ) {

//        printf("this is not for real boys");
        PotentialArray potVals(ncalls, PotentialVector(num_walkers, 52.0));
        return potVals;

        }

#endif

PyObject *RynLib_callPot(PyObject* self, PyObject* args ) {

    PyObject* atoms;
    PyObject* coords;
    if ( !PyArg_ParseTuple(args, "OO", &atoms, &coords) ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;
    Coordinates walker_coords = _getWalkerCoords(raw_data, 0, num_atoms);
    double pot;
//    try {
    pot = _doopAPot(walker_coords, mattsAtoms);
//    } catch(...) {
//        return NULL;
//    }

    PyObject *potVal = Py_BuildValue("f", pot);
    return potVal;

}

PyObject *RynLib_callPotVec( PyObject* self, PyObject* args ) {
    // vector version of callPot

    PyObject* atoms;
    PyObject* coords;
    if ( !PyArg_ParseTuple(args, "OO", &atoms, &coords) ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    if (PyErr_Occurred()) return NULL;
    Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    Py_ssize_t ncalls = PyObject_Length(coords);
    if (PyErr_Occurred()) return NULL;

    PyObject *shape = PyObject_GetAttrString(coords, "shape");
    if (shape == NULL) return NULL;
    PyObject *num_obj = PyTuple_GetItem(shape, 1);
    if (num_obj == NULL) return NULL;
    Py_ssize_t num_walkers = _FromInt(num_obj);
    if (PyErr_Occurred()) return NULL;

    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;

    int world_size, world_rank;
    _mpiInit(&world_size, &world_rank);
    // Get vector of values from MPI call

    PotentialArray pot_vals = _mpiGetPot(
                raw_data, mattsAtoms,
                ncalls,
                num_walkers,
                num_atoms,
                world_size,
                world_rank
                );

    if ( world_rank == 0 ){
        return _fillNumPyArray(pot_vals, ncalls, num_walkers);
    } else {
        Py_RETURN_NONE;
    }

}

PyObject *RynLib_testPot( PyObject* self, PyObject* args ) {

    PyObject *hello;

    hello = Py_BuildValue("f", 50.2);
    return hello;

}

// New design that will make use of the WalkerPropagator object I'm setting up
/*
PyObject *RynLib_getWalkers( PyObject* self, PyObject* args ) {

    PyObject* cores;
    if ( !PyArg_ParseTuple(args, "O", &cores) ) return NULL;

    Coordinates walker_positions = _mpiGetWalkersFromNodes(

    );

}
 */

RawPotentialBuffer WalkerPropagator::get_pots() {

    auto pots = (RawPotentialBuffer) malloc(prop_steps*walkers.size());

    for (int j = 0; j < walkers.size(); j++) { // number of walkers we're holding onto
        for (int i = 0; i < prop_steps; i++) { // number of steps per walker
            // propagate the j-th walker forward
            for (int n = 0; n < atoms.size(); n++ ) {
                for (int m = 0; m < 3; m++ ) {
                    walkers[j][n][m] += distributions[m](engine);
                }
            }
            // call potential on newly moved walker and stick this in pots
            pots[j*prop_steps + i] = _doopAPot(walkers[j], atoms);
        }
    }

    return pots;
}

// MPI COMMUNICATION METHODS
PyObject *RynLib_initializeMPI(PyObject *self, PyObject *args) {

    PyObject *hello;
    int world_size, world_rank;
    _mpiInit(&world_size, &world_rank);
    hello = Py_BuildValue("(ii)", world_rank, world_size);
    return hello;

}

PyObject *RynLib_finalizeMPI(PyObject *self, PyObject *args) {

    _mpiFinalize();
    Py_RETURN_NONE;

}

PyObject *RynLib_holdMyPI( PyObject* self, PyObject* args ) {

    _mpiBarrier();
    Py_RETURN_NONE;

}


// PYTHON WRAPPER EXPORT

static PyMethodDef RynLibMethods[] = {
    {"rynaLovesDMC", RynLib_callPot, METH_VARARGS, "calls entos on a single walker"},
    {"rynaLovesDMCLots", RynLib_callPotVec, METH_VARARGS, "will someday call entos on a vector of walkers"},
    {"rynaSaysYo", RynLib_testPot, METH_VARARGS, "a test flat potential for debugging"},
    {"giveMePI", RynLib_initializeMPI, METH_VARARGS, "calls Init and returns the processor rank"},
    {"noMorePI", RynLib_finalizeMPI, METH_VARARGS, "calls Finalize in a safe fashion (can be done more than once)"},
    {"holdMyPI", RynLib_holdMyPI, METH_VARARGS, "calls Barrier"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION > 2

const char RynLib_doc[] = "RynLib is for Ryna Dorisii";
static struct PyModuleDef RynLibModule = {
    PyModuleDef_HEAD_INIT,
    "RynLib",   /* name of module */
    RynLib_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    RynLibMethods
};

PyMODINIT_FUNC PyInit_RynLib(void)
{
    return PyModule_Create(&RynLibModule);
}
#else

PyMODINIT_FUNC initRynLib(void)
{
    (void) Py_InitModule("RynLib", RynLibMethods);
}

#endif

#endif