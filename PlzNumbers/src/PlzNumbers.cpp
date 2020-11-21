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

            ExtraBools extra_bools;
            ExtraInts extra_ints;
            ExtraFloats extra_floats;

            PyObject *iterator, *item;

            iterator = PyObject_GetIter(ext_bool);
            if (iterator == NULL) {
                throw std::runtime_error("Iteration error");
            }
            while ((item = PyIter_Next(iterator))) {
                extra_bools.push_back(_FromBool(item));
                Py_DECREF(item);
            }
            Py_DECREF(iterator);
            if (PyErr_Occurred()) {
                throw std::runtime_error("Iteration error");
            }

            iterator = PyObject_GetIter(ext_int);
            if (iterator == NULL) {
                throw std::runtime_error("Iteration error");
            }
            while ((item = PyIter_Next(iterator))) {
                extra_ints.push_back(_FromInt(item));
                Py_DECREF(item);
            }
            Py_DECREF(iterator);
            if (PyErr_Occurred()) {
                throw std::runtime_error("Iteration error");
            }

            iterator = PyObject_GetIter(ext_float);
            if (iterator == NULL) {
                throw std::runtime_error("Iteration error");
            }
            while ((item = PyIter_Next(iterator))) {
                extra_floats.push_back(_FromFloat(item));
                Py_DECREF(item);
            }
            Py_DECREF(iterator);
            if (PyErr_Occurred()) {
                throw std::runtime_error("Iteration error");
            }

            PyObject *str = NULL;
            std::string bad_walkers_file = _GetPyString(walkers_file, str);
            Py_XDECREF(str);

            ExtraArgs args{
                    bad_walkers_file,
                    err_val,
                    debug_print,
                    retries,

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
                bool use_TBB
        ) {
            ThreadingMode mode = ThreadingMode::SERIAL;
            if (vectorized_potential) {
                mode = ThreadingMode::VECTORIZED;
            } else if (use_openMP) {
                mode = ThreadingMode::OpenMP;
            } else if (use_TBB) {
                mode = ThreadingMode::TBB;
            }

            const char *func_name = "_potential";
            PotentialFunction pot_func = NULL;
            FlatPotentialFunction flat_pot_func = NULL;
            VectorizedPotentialFunction vec_pot_func = NULL;
            VectorizedFlatPotentialFunction vec_flat_pot_func = NULL;
            if (vectorized_potential) {
                if (raw_array_pot) {
                    vec_flat_pot_func = (VectorizedFlatPotentialFunction) PyCapsule_GetPointer(capsule, func_name);
                    if (vec_flat_pot_func == NULL) {
                        throw std::runtime_error("Capsule error");
                    }
                } else {
                    vec_pot_func = (VectorizedPotentialFunction) PyCapsule_GetPointer(capsule, func_name);
                    if (vec_pot_func == NULL) {
                        throw std::runtime_error("Capsule error");
                    }
                }
            } else {
                if (raw_array_pot) {
                    flat_pot_func = (FlatPotentialFunction) PyCapsule_GetPointer(capsule, func_name);
                    if (flat_pot_func == NULL) {
                        throw std::runtime_error("Capsule error");
                    }
                } else {
                    pot_func = (PotentialFunction) PyCapsule_GetPointer(capsule, func_name);
                    if (pot_func == NULL) {
                        throw std::runtime_error("Capsule error");
                    }
                }

            }

            PotentialApplier pot_fun(pot_func, flat_pot_func, vec_pot_func, vec_flat_pot_func);
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
    int use_openMP, use_TBB, retries;

    if ( !PyArg_ParseTuple(
            args,
            "OOOOOdppppOpp",
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
            &use_TBB
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
                use_TBB
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