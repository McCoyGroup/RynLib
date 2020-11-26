//
// The layer between our code and python
// explicitly tries to avoid doing much work
//

#include "PlzNumbers.hpp"
#include "PyAllUp.hpp"
#include <stdexcept>
//#include <memory>


namespace rynlib {

    namespace common {
        bool DEBUG_PRINT=false;
        bool debug_print() {
            return DEBUG_PRINT;
        }
        void set_debug_print(bool db) {
            DEBUG_PRINT = db;
        }
    }
    using namespace common;
    using namespace python;
    namespace PlzNumbers {

        ThreadingHandler load_caller(
                PyObject *capsule,
                CallerParameters& parameters
        ) {
            ThreadingMode mode = parameters.threading_mode();
            PotentialApplier pot_fun(capsule, parameters);
            return {pot_fun, mode};
        }

        CoordsManager load_coords(PyObject *coords, PyObject *atoms) {
            try {
                if ( debug_print() ) {
                    printf("  > loading atoms\n");
                }
                auto mattsAtoms = from_python_iterable<std::string>(atoms);
                if ( rynlib::common::debug_print() ) {
                    printf("  > loading shape\n");
                }
                std::vector<size_t> shape = numpy_shape<size_t>(coords);
                if ( rynlib::common::debug_print() ) {
                    printf("  > getting pointer to data\n");
                }
                auto raw_data = get_numpy_data<Real_t>(coords);

                return {raw_data, mattsAtoms, shape};
            } catch (std::exception& e) {
                std::string err_msg = "Failed to load coordinates: ";
                err_msg += e.what();
                if ( rynlib::common::debug_print() ) {
                    printf("  > ERROR: %s\n", err_msg.c_str());
                }
                throw std::runtime_error(err_msg);
            }

        }

        CallerParameters load_parameters(PyObject* atoms, PyObject* parameters) {
            try {
                return {atoms, parameters};
            } catch (std::exception& e) {
                std::string err_msg = "Failed to load parameters: ";
                err_msg += e.what();
                if ( rynlib::common::debug_print() ) {
                    printf("  > ERROR: %s\n", err_msg.c_str());
                }
                throw std::runtime_error(err_msg);
            }
        }

        MPIManager load_mpi(PyObject* mpi) {
            try {
                auto manager = MPIManager(mpi);
                return manager;
            } catch (std::exception& e) {
                std::string err_msg = "Failed to load MPI: ";
                err_msg += e.what();
                if ( rynlib::common::debug_print() ) {
                    printf("  > ERROR: %s\n", err_msg.c_str());
                }
                throw std::runtime_error(err_msg);
            }
        }

        PotentialCaller load_evaluator(CoordsManager& coord_data, MPIManager& mpi, ThreadingHandler& caller) {
            try {
                return {coord_data, mpi, caller};
            } catch (std::exception& e) {
                std::string err_msg = "Failed to load evaluator: ";
                err_msg += e.what();
                if ( rynlib::common::debug_print() ) {
                    printf("  > ERROR: %s\n", err_msg.c_str());
                }
                throw std::runtime_error(err_msg);
            }
//            rynlib::PlzNumbers::PotentialCaller evaluator(
//                    coord_data,
//                    mpi,
//                    caller
//            );
        }

    }

}

PyObject *PlzNumbers_callPotVec(PyObject* self, PyObject* args ) {

    int debug;
    PyObject* coords, *atoms, *pot_function, *parameters, *manager;

//    auto garb = rynlib::python::get_python_repr(args);
//    printf("  >>> 2.1 waaat %s\n", garb.c_str());

    int passed = PyArg_ParseTuple(
            args,
            "pOOOOO",
            &debug,
            &coords,
            &atoms,
            &pot_function,
            &parameters,
            &manager
    );
    if ( !passed ) return NULL;

    try {

        set_debug_print(debug);
        plzffi::set_debug_print(debug);
        rynlib::python::pyadeeb.set_debug_print(debug);

        if ( rynlib::common::debug_print() ) {
            printf("Loading coords/atom data from PyObjects...\n");
        }
        auto coord_data = rynlib::PlzNumbers::load_coords(
                coords,
                atoms
        );
        if (PyErr_Occurred()) { throw std::runtime_error("failed to load coords..."); }

        if ( rynlib::common::debug_print() ) {
            printf("Loading parameters from PyObjects...\n");
        }
        auto params = rynlib::PlzNumbers::load_parameters(coords, parameters);
        if (PyErr_Occurred()) { throw std::runtime_error("failed to load parameters..."); }

        if ( rynlib::common::debug_print() ) {
            printf("Loading MPI...\n");
        }
        auto mpi = rynlib::PlzNumbers::load_mpi(manager);
        if (PyErr_Occurred()) { throw std::runtime_error("failed to load MPI..."); }

        if ( rynlib::common::debug_print() ) {
            printf("Loading caller...\n");
        }
        auto caller = rynlib::PlzNumbers::load_caller(pot_function, params);
        auto evaluator = rynlib::PlzNumbers::load_evaluator(coord_data, mpi, caller);
        if (PyErr_Occurred()) { throw std::runtime_error("failed to load caller..."); }

        // if ( mpi.is_main() && caller.call_parameters().debug()) {
        if (rynlib::common::debug_print()) {
            printf("Calling into evaluator...\n");
        }
        auto pot_vals = evaluator.get_pot();
        if (PyErr_Occurred()) { throw std::runtime_error("failure in evaluation..."); }

        if (mpi.is_main()) {

//            printf("  > huh? %f...\n", pot_vals.vector()[0]);

//            if (mpi.is_main() && caller.call_parameters().debug()) {
            if (rynlib::common::debug_print()) {
                printf("  > constructing NumPy array...\n");
            }
            auto base_pot = rynlib::python::numpy_from_data<Real_t>(pot_vals.data(), pot_vals.get_shape());
            if (PyErr_Occurred()) { throw std::runtime_error("failed to construct array..."); }
            // I think I don't need to incref this, but we may need to revisit that thought

            auto pot_obj = rynlib::python::numpy_copy_array(base_pot);
            Py_XDECREF(base_pot);

            return pot_obj;
        } else {
            if (rynlib::common::debug_print()) {
                printf("     > done on subsidiary thread...\n");
            }
            Py_RETURN_NONE;
        }

    } catch (std::exception &e) {
        if (!PyErr_Occurred()) {
            std::string msg = "In C++ caller: ";
            msg += e.what();
            PyErr_SetString(PyExc_SystemError, msg.c_str());
        }
        return NULL;
    }

}

// PYTHON WRAPPER EXPORT (Python 3 only)

static PyMethodDef PlzNumbersMethods[] = {
        {"rynaLovesPootsLots", PlzNumbers_callPotVec, METH_VARARGS, "calls a potential on a vector of walkers"},
        {NULL, NULL, 0, NULL}
};

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