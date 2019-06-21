#include "RynLib.h"

typedef std::vector<double> Point;
typedef Point PotentialVector;
typedef std::vector< Point > Coordinates;
typedef Coordinates PotentialArray;
typedef std::vector< Coordinates > Configurations;
typedef std::string Name;
typedef std::vector<std::string> Names;

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

double _doopAPot(const Coordinates &walker_coords, const Names &atoms) {
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
    Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

    // Assumes number of walkers X number of atoms X 3
    double* raw_data = _GetDoubleDataArray(coords);
    if (raw_data == NULL) return NULL;
    Coordinates walker_coords = _getWalkerCoords(raw_data, 0, num_atoms);
    double pot = _doopAPot(walker_coords, mattsAtoms);

    PyObject *potVal = Py_BuildValue("f", pot);
    return potVal;

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

PotentialArray _mpiGetPot(
        double* raw_data,
        const Names &atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank
        ) {

    // create a buffer for the walkers to be fed into MPI
    auto walker_buf = (double*) malloc(ncalls*num_atoms*3*sizeof(double));

    // Scatter data buffer to processors
    MPI_Scatter(
                raw_data,  // raw data buffer to chunk up
                ncalls*3*num_atoms, // three coordinates per atom per num_atoms per walker
                MPI_DOUBLE, // coordinates stored as doubles
                walker_buf, // raw array to write into
                ncalls*3*num_atoms, // single energy
                MPI_DOUBLE, // energy returned as doubles
                0, // root caller
                MPI_COMM_WORLD // communicator handle
                );

    Coordinates walker_coords (num_walkers, Point(3));

    auto pot_buf = (double*) malloc(ncalls*world_size*sizeof(double)); // receive buffer

    PotentialVector pots(ncalls);
    for (int i = 0; i < ncalls; i++) {
        walker_coords = _getWalkerCoords(walker_buf, i, num_atoms);
        pots[i] = _doopAPot(walker_coords, atoms);
    }

    MPI_Gather(pots.data(), ncalls, MPI_DOUBLE, pot_buf, ncalls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(walker_buf);

    // convert double* to std::vector<double>
    PotentialArray potVals(ncalls, PotentialVector(world_size));
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
                ncalls, num_walkers, num_atoms,
                world_size, world_rank
                );

    if ( world_rank == 0 ){
        PyObject *array_module = PyImport_ImportModule("numpy");
        if (array_module == NULL) return NULL;
        PyObject *builder = PyObject_GetAttrString(array_module, "zeros");
        Py_XDECREF(array_module);
        if (builder == NULL) return NULL;
        PyObject *dims = Py_BuildValue("((ii))", ncalls, num_walkers);
        Py_XDECREF(builder);
        if (dims == NULL) return NULL;
        PyObject *kw = Py_BuildValue("{s:s}", "dtype", "float");
        if (kw == NULL) return NULL;
        PyObject *pot = PyObject_Call(builder, dims, kw);
        Py_XDECREF(kw);
        Py_XDECREF(dims);
        if (pot == NULL) return NULL;
        double* data = _GetDoubleDataArray(pot);

        for (int i = 0; i < ncalls; i++) {
            memcpy(data + num_walkers * i, pot_vals[i].data(), sizeof(double) * num_walkers);
        };

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

PyObject *RynLib_holdMyPI( PyObject* self, PyObject* args ) {

    _mpiBarrier();
    Py_RETURN_NONE;

}

static PyMethodDef RynLibMethods[] = {
    {"rynaLovesDMC", RynLib_callPot, METH_VARARGS, "calls entos on a single walker"},
    {"rynaLovesDMCLots", RynLib_callPotVec, METH_VARARGS, "will someday call entos on a vector of walkers"},
    {"rynaSaysYo", RynLib_testPot, METH_VARARGS, "a test flat potential for debugging"},
    {"giveMePI", RynLib_giveMePI, METH_VARARGS, "calls Init and returns the processor rank"},
    {"noMorePI", RynLib_noMorePI, METH_VARARGS, "calls Finalize in a safe fashion (can be done more than once)"},
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
