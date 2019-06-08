#include "RynLib.h"
#include <random>

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
#else
const char *_GetPyString( PyObject* s ) {
    return PyString_AsString(s);
}
const char *_GetPyString( PyObject* s, PyObject *pyStr ) {
    // just to unify the 2/3 interface
    return _GetPyString( s );
    }
#endif

std::vector<std::string> _getAtomTypes( PyObject* atoms, Py_ssize_t num_atoms ) {

    std::vector<std::string> mattsAtoms(num_atoms);
    for (int i = 0; i<num_atoms; i++) {
        PyObject* atom = PyList_GetItem(atoms, i);
        PyObject* pyStr = NULL;
        const char* atomStr = _GetPyString(atom, pyStr);
        std::string atomString = atomStr;
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

inline int int3d(int i, int j, int k, int m, int l) {
    return (m*l) * i + (l*j) + k;
}

std::vector< std::vector<double> > _getWalkerCoords(double* raw_data, int i, Py_ssize_t num_atoms) {
    std::vector< std::vector<double> > walker_coords(num_atoms, std::vector<double>(3));
    for (int j = 0; j<num_atoms; j++) {
        for (int k = 0; k<3; k++) {
            walker_coords[j][k] = raw_data[int3d(i, j, k, num_atoms, 3)];
        }
    };
    return walker_coords;
}

void _printOutWalkerStuff( std::vector< std::vector<double> > walker_coords ) {
    printf("This walker was bad: ( ");
    for ( int i = 0; i < walker_coords.size(); i++) {
        printf("(%f, %f, %f)", walker_coords[i][0], walker_coords[i][1], walker_coords[i][2]);
        if ( i < walker_coords.size()-1 ) {
            printf(", ");
        }
    }
    printf(" )\n");
}

double _doopAPot(std::vector< std::vector<double> > walker_coords, std::vector<std::string> atoms) {
    double pot;
    try {
        pot = MillerGroup_entosPotential(walker_coords, atoms);
    } catch (int e) {
        _printOutWalkerStuff(walker_coords);
        pot = 1.0e9;
    } catch (const char* e) {
        _printOutWalkerStuff(walker_coords);
        pot = 1.0e9;
    } catch (...) {
        _printOutWalkerStuff(walker_coords);
        pot = 1.0e9;
    }

    return pot;
};

PyObject *RynLib_callPot(PyObject* self, PyObject* args ) {

    PyObject* atoms;
    PyObject* coords;
    if ( !PyArg_ParseTuple(args, "OO", &atoms, &coords) ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    std::vector<std::string> mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;
    std::vector< std::vector<double> > walker_coords = _getWalkerCoords(raw_data, 0, num_atoms);
    double pot = _doopAPot(walker_coords, mattsAtoms);

    PyObject *potVal = Py_BuildValue("f", pot);
    return potVal;

}

#ifdef IM_A_REAL_BOY

void _mpiInit(int* world_size, int* world_rank) {
    // Initialize MPI state
    int did_i_do_good_pops = 0;
    MPI_Initialized(&did_i_do_good_pops); // need to check if we called Init once already
    // printf("How'd I do? %d\n", did_i_do_good_pops);
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

std::vector<std::vector<double> > _mpiGetPot(
        double* raw_data,
        std::vector<std::string> atoms,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank,
        int ncalls
        ) {

    // create a buffer for the walkers to be fed into MPI
    double* walker_buf = (double*) malloc(num_atoms*3*sizeof(double));
    // Scatter data buffer to processors
    MPI_Scatter(
                raw_data,  // raw data buffer to chunk up
                3*num_atoms, // three coordinates per atom per num_atoms per walker
                MPI_DOUBLE, // coordinates stored as doubles
                walker_buf, // raw array to write into
                3*num_atoms, // single energy
                MPI_DOUBLE, // energy returned as doubles
                0, // root caller
                MPI_COMM_WORLD // communicator handle
                );

    std::vector< std::vector<double> > walker_coords = _getWalkerCoords(walker_buf, 0, num_atoms);
    // this is basically a hold-over from a previous implementation where our
    // double* pot_buf;
    // if (world_rank == 0) {
    double* pot_buf = (double*) malloc(ncalls*world_size*sizeof(double)); // receive buffer
    //};

    std::vector<double> pots(ncalls);
    for (int i = 0; i < ncalls; i++) {

        pots[i] = _doopAPot(walker_coords, atoms);
        // TODO: Insert code to perturb walkers

    }


    MPI_Gather(pots.data(), ncalls, MPI_DOUBLE, pot_buf, ncalls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(walker_buf);

    // convert double* to std::vector<double>
    std::vector<std::vector<double> > potVals(ncalls, std::vector<double>(world_size));
    if( world_rank == 0 ) {
        // this is effectively column ordered at this point I think...
        for (size_t j = 0; j < world_size; j++) {
            for (size_t i = 0; i < ncalls; i++) {
                potVals[i][j] = pot_buf[j*ncalls + i];
            }
        }
    }
    free(pot_buf);

    return potVals;
}

#else

void _mpiInit(int* world_size, int* world_rank) {
    world_size = 0;
    world_rank = 0;
}

void _mpiFinalize() {
    // boop
}

std::vector<std::vector<double> > _mpiGetPot(
        double* raw_data,
        std::vector<std::string> atoms,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank,
        int ncalls
        ) {

        printf("this is not for real boys");
        std::vector<d<std::vector<double> > potVals(ncalls, std::vector<double>(num_walkers));
        return potVals;

        }

#endif

PyObject *RynLib_callPotVec( PyObject* self, PyObject* args ) {
    // vector version of callPot

    PyObject* atoms;
    PyObject* coords;
    int ncalls;
    if ( !PyArg_ParseTuple(args, "OOi", &atoms, &coords, &ncalls) ) return NULL;

    // Assumes we get n atom type names
    Py_ssize_t num_atoms = PyObject_Length(atoms);
    std::vector<std::string> mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    Py_ssize_t num_walkers = PyObject_Length(coords);
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;
    int world_size, world_rank;
    _mpiInit(&world_size, &world_rank);
    // Get vector of values from MPI call
    std::vector<std::vector<double> > pot_vals = _mpiGetPot(
                raw_data, mattsAtoms, num_walkers, num_atoms,
                world_size, world_rank, ncalls
                );

    if ( world_rank == 0 ){
        // handle return to python sans error checking because laziness
        PyObject *array_module = PyImport_ImportModule("numpy");
        PyObject *builder = PyObject_GetAttrString(array_module, "zeros");
        // issue might be that I need to pass dtype = float...
        PyObject *dims = Py_BuildValue("((ii))", ncalls, num_walkers);
        PyObject *kw = Py_BuildValue("{s:s}", "dtype", "float");
        PyObject *pot = PyObject_Call(builder, dims, kw);
        Py_XDECREF(dims);
        Py_XDECREF(builder);
        Py_XDECREF(array_module);
        Py_XDECREF(kw);
        double* data = _GetDoubleDataArray(pot);
        memcpy(data, pot_vals.data(), sizeof(double)*num_walkers*ncalls);

        return pot;
    } else {
        Py_RETURN_NONE;
    }
}

PyObject *RynLib_testPot( PyObject* self, PyObject* args ) {

    PyObject *hello;

    hello = Py_BuildValue("f", 50.2);
    return hello;

}

PyObject *RynLib_giveMePI( PyObject* self, PyObject* args ) {

    PyObject *hello;
    int world_size, world_rank;
    _mpiInit(&world_size, &world_rank);
    hello = Py_BuildValue("(ii)", world_rank, world_size);
    return hello;

}

PyObject *RynLib_noMorePI( PyObject* self, PyObject* args ) {

    _mpiFinalize();
    Py_RETURN_NONE;

}

static PyMethodDef RynLibMethods[] = {
    {"rynaLovesDMC", RynLib_callPot, METH_VARARGS, "calls entos on a single walker"},
    {"rynaLovesDMCLots", RynLib_callPotVec, METH_VARARGS, "will someday call entos on a vector of walkers"},
    {"rynaSaysYo", RynLib_testPot, METH_VARARGS, "a test flat potential for debugging"},
    {"giveMePI", RynLib_giveMePI, METH_VARARGS, "calls Init and returns the processor rank"},
    {"noMorePI", RynLib_noMorePI, METH_VARARGS, "calls Finalize in a safe fashion (can be done more than once)"},
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
