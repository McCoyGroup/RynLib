//
// The layer between our code and python
// explicitly tries to avoid doing much work
//

#include "PlzNumbers.hpp"
#include "PyAllUp.hpp"
#include <stdexcept>
//#include <memory>

namespace rynlib {
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

        CoordsManager load_coords(
                PyObject *coords,
                PyObject *atoms
        ) {

            auto mattsAtoms = from_python_iterable<std::string>(atoms);
            std::vector<size_t> shape = numpy_shape<size_t>(coords);
            auto raw_data = get_numpy_data<Real_t >(coords);

            return {raw_data, mattsAtoms, shape};

        }

    }
}

PyObject *PlzNumbers_callPotVec(PyObject* self, PyObject* args ) {

    PyObject* coords, *atoms, *pot_function, *parameters, *manager;

//    auto garb = rynlib::python::get_python_repr(args);
//    printf("  >>> 2.1 waaat %s\n", garb.c_str());

    int passed = PyArg_ParseTuple(
            args,
            "OOOOO",
            &coords,
            &atoms,
            &pot_function,
            &parameters,
            &manager
    );
    if ( !passed ) return NULL;

    try {

        auto coord_data = rynlib::PlzNumbers::load_coords(
                coords,
                atoms
        );
        rynlib::PlzNumbers::CallerParameters params(atoms, parameters);
        auto caller = rynlib::PlzNumbers::load_caller(pot_function, params);
        rynlib::PlzNumbers::MPIManager mpi(manager);
        rynlib::PlzNumbers::PotentialCaller evaluator(
                coord_data,
                mpi,
                caller
        );

        auto wat = caller.call_parameters();

        auto pot_vals = evaluator.get_pot();

        if (mpi.is_main()) {
            auto pot_obj = rynlib::python::numpy_from_data<Real_t>(pot_vals.data(), pot_vals.get_shape());
            // I think I don't need to incref this, but we may need to revisit that thought
            return pot_obj;
        } else {
            Py_RETURN_NONE;
        }

    } catch (std::exception &e) {
        // maybe I want to set a message -> just here to protect us against segfaults and shit...
        if (!PyErr_Occurred()) {
            std::string msg = "in C++ caller: ";
            msg += e.what();
//            printf("%s", msg.c_str());
            PyErr_SetString(
                    PyExc_SystemError,
                    msg.c_str()
                    );
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