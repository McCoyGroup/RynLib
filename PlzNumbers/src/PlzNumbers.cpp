//
// The layer between our code and python
// explicitly tries to avoid doing much work
//

#include "PlzNumbers.hpp"
#include "PyAllUp.hpp"
#include <stdexcept>

namespace rynlib {
    using namespace common;
    using namespace python;
    namespace PlzNumbers {

        ExtraArgs load_args(
                PyObject *extra_args,
                PyObject *walkers_file, Real_t err_val, int retries,
                bool debug_print
        ) {

            PyObject* ext_bool = PyTuple_GetItem(extra_args, 0);
            PyObject* ext_int = PyTuple_GetItem(extra_args, 1);
            PyObject* ext_float = PyTuple_GetItem(extra_args, 2);

            auto extra_bools = from_python_iterable<bool>(ext_bool);
            auto extra_ints = from_python_iterable<int>(ext_int);
            auto extra_floats = from_python_iterable<double>(ext_float);
            auto bad_walkers_file = from_python<std::string>(walkers_file);

            ExtraArgs args{
                    bad_walkers_file,
                    err_val,
                    debug_print,
                    retries,

                    extra_args,
                    extra_bools,
                    extra_ints,
                    extra_floats
            };

            return args;

        }

        ThreadingHandler load_caller(
                PyObject *capsule,
                bool raw_array_pot,
                bool vectorized_potential,
                bool use_openMP,
                bool use_TBB,
                bool python_potential
        ) {
            ThreadingMode mode = ThreadingMode::SERIAL;
            if (vectorized_potential) {
                mode = ThreadingMode::VECTORIZED;
            } else if (use_openMP) {
                mode = ThreadingMode::OpenMP;
            } else if (use_TBB) {
                mode = ThreadingMode::TBB;
            }

            PotentialFunction pot_func = NULL;
            FlatPotentialFunction flat_pot_func = NULL;
            VectorizedPotentialFunction vec_pot_func = NULL;
            VectorizedFlatPotentialFunction vec_flat_pot_func = NULL;
            if (!python_potential) {
                const char *func_name = "_potential";
                if (vectorized_potential) {
                    if (raw_array_pot) {
                        vec_flat_pot_func = from_python_capsule<VectorizedFlatPotentialFunction>(capsule, func_name);
                    } else {
                        vec_pot_func = from_python_capsule<VectorizedPotentialFunction>(capsule, func_name);
                    }
                } else {
                    if (raw_array_pot) {
                        flat_pot_func = from_python_capsule<FlatPotentialFunction>(capsule, func_name);
                    } else {
                        pot_func = from_python_capsule<PotentialFunction>(capsule, func_name);
                    }

                }
            }

            PotentialApplier pot_fun(capsule, pot_func, flat_pot_func, vec_pot_func, vec_flat_pot_func, python_potential);
            return {pot_fun, mode};

        }

        CoordsManager load_coords(
                PyObject *coords,
                PyObject *atoms
        ) {

            // Assumes we get n atom type names
            Py_ssize_t num_atoms = PyObject_Length(atoms);
            Names mattsAtoms = _getAtomTypes(atoms, num_atoms);

            // Assumes number of walkers X number of atoms X 3
            double* raw_data = _GetDoubleDataArray(coords);
            if (raw_data == NULL) {
                throw std::runtime_error("NumPy issues");
            }

            // we'll assume we have number of walkers X ncalls X number of atoms X 3
            PyObject *shape = PyObject_GetAttrString(coords, "shape");
            if (shape == NULL) {
                throw std::runtime_error("NumPy issues");
            }
            PyObject *ncalls_obj = PyTuple_GetItem(shape, 1);
            if (ncalls_obj == NULL) {
                throw std::runtime_error("NumPy issues");
            }
            Py_ssize_t ncalls = _FromInt(ncalls_obj);
            if (PyErr_Occurred()) {
                throw std::runtime_error("NumPy issues");
            }
            PyObject *num_walkers_obj = PyTuple_GetItem(shape, 0);
            if (num_walkers_obj == NULL) {
                throw std::runtime_error("NumPy issues");
            }
            Py_ssize_t num_walkers = _FromInt(num_walkers_obj);
            if (PyErr_Occurred()) {
                throw std::runtime_error("NumPy issues");
            }

            return {
                raw_data,
                mattsAtoms,
                {static_cast<size_t >(ncalls), num_walkers} // CLion said I had to...
            };

        }

        PyObject* recompose_NumPy_array(
                CoordsManager coords,
                PotValsManager pot_vals
                ) {

            return _fillNumPyArray(
                    pot_vals.data(),
                    coords.get_shape()[0],
                    coords.get_shape()[1]
                    );

        }
    }
}


PyObject *PlzNumbers_callPot(PyObject* self, PyObject* args ) {

    PyObject* coords, *atoms, *pot_function, *extra_args, *bad_walkers_file;
    double err_val;
    int raw_array_pot, vectorized_potential, debug_print;
    PyObject* manager;
    int use_openMP, use_TBB, retries, python_potential;

    if ( !PyArg_ParseTuple(
            args,
            "OOOOOdppppOppp",
            &coords,
            &atoms,
            &pot_function,
            &extra_args,
            &bad_walkers_file,
            &err_val,
            &raw_array_pot,
            &vectorized_potential,
            &debug_print,
            &retries,
            &manager,
            &use_openMP,
            &use_TBB,
            &python_potential
    )
            ) return NULL;

    try {

        auto coord_data = rynlib::PlzNumbers::load_coords(
                coords,
                atoms
        );

        auto arg_list = rynlib::PlzNumbers::load_args(
                extra_args,
                bad_walkers_file, err_val, retries,
                debug_print
        );

        auto caller = rynlib::PlzNumbers::load_caller(
                pot_function,
                raw_array_pot,
                vectorized_potential,
                use_openMP,
                use_TBB,
                python_potential
                );

        rynlib::PlzNumbers::MPIManager mpi(manager);

        rynlib::PlzNumbers::PotentialCaller evaluator(
                coord_data,
                mpi,
                caller,
                arg_list
        );

        auto pot_vals = evaluator.get_pot();

        PyObject* pot_obj = rynlib::PlzNumbers::recompose_NumPy_array(
                coord_data,
                pot_vals
        );


        return pot_obj;

    } catch (std::exception &e) {
        // maybe I want to set a message -> just here to protect us against segfaults and shit...
        return NULL;

    }

}

// PYTHON WRAPPER EXPORT

static PyMethodDef PlzNumbersMethods[] = {
        {"rynaLovesPoots", PlzNumbers_callPot, METH_VARARGS, "calls a potential on a single walker"},
        {"rynaLovesPootsLots", PlzNumbers_callPotVec, METH_VARARGS, "calls a potential on a vector of walkers"},
        {"rynaLovesPyPootsLots", PlzNumbers_callPyPotVec, METH_VARARGS, "calls a _python_ potential on a vector of walkers"},
        {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION > 2

const char PlzNumbers_doc[] = "PlzNumbers manages the calling of a potential at the C++ level";
static struct PyModuleDef PlzNumbersModule = {
        PyModuleDef_HEAD_INIT,
        "PlzNumbers",   /* name of module */
        PlzNumbers_doc, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        PlzNumbersMethods
};

PyMODINIT_FUNC PyInit_PlzNumbers(void)
{
    return PyModule_Create(&PlzNumbersModule);
}
#else

PyMODINIT_FUNC initPlzNumbers(void)
{
    (void) Py_InitModule("PlzNumbers", PlzNumbersMethods);
}

#endif