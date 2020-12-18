#include "MPIManager.hpp"
#include "PyAllUp.hpp"
#include <stdexcept>

namespace rynlib {

    using namespace common;
    namespace PlzNumbers {

        void MPIManager::init() {

            if (no_mpi()) {
                mpi_world_size = 1;
                mpi_world_rank = 0;
            } else {
                mpi_world_size = rynlib::python::get_python_attr<int>(mpi_manager, "world_size");
                mpi_world_rank = rynlib::python::get_python_attr<int>(mpi_manager, "world_rank");
            }

        }

        CoordsManager MPIManager::scatter_walkers(CoordsManager &coords) {

            if (no_mpi()) return coords; // short circuit

            // The way this works is that we start with an array of data that looks like (ncalls, num_walkers, *walker_shape)
            // Then we have m cores such that num_walkers_per_core = num_walkers / m
            //
            // We pass these in to MPI and allow them to get distributed out as blocks of ncalls * num_walkers_per_core walkers
            // to a given core, which calculates the potential over all of them and then returns that
            //
            // At the end we have a potential array that is m * (ncalls * num_walkers_per_core) walkers and we need to make this
            // back into the clean (ncalls, num_walkers) array we expect in the end
            auto ncalls = coords.num_calls();
            auto num_walkers = coords.num_walkers();
            // we're gonna assume the former is divisible by the latter on world_rank == 0
            // and that it's just plain `num_walkers` on every other world_rank
            auto num_walkers_per_core = (num_walkers / world_size());
            if (world_rank() > 0) {
                // means we're only feeding in num_walkers because we're not on world_rank == 0
                num_walkers_per_core = num_walkers;
            } else if (num_walkers % world_size()) {
                throw std::runtime_error("Number of walkers not divisible by number of MPI processes");
            }

            // create a buffer for the walkers to be fed into MPI
            auto num_atoms = coords.num_atoms();
            auto walker_cnum = num_atoms*3;
            auto walkers_to_core = ncalls * num_walkers_per_core;

            auto walker_buf = (RawWalkerBuffer) malloc(walkers_to_core * walker_cnum * sizeof(Real_t));

            // Scatter data buffer to processors
            PyObject* scatter = PyObject_GetAttrString(mpi_manager, "scatter");
            auto scatter_walkers = (ScatterFunction) PyCapsule_GetPointer(scatter, "Dumpi._SCATTER_WALKERS");
            scatter_walkers(
                    mpi_manager,
                    coords.data(),  // raw data buffer to chunk up
                    walkers_to_core,
                    walker_cnum, // three coordinates per atom per num_atoms per walker
                    walker_buf // raw array to write into
            );
            Py_XDECREF(scatter);

            // Now we build a new CoordsManager

            auto atoms = coords.get_atoms();
            auto shp = coords.get_shape();
            shp[0] = num_walkers_per_core;
            return CoordsManager(walker_buf, atoms, shp);
        }

        PotValsManager MPIManager::gather_potentials(
                CoordsManager &coords,
                PotValsManager &pots
        ) {

            if (no_mpi()) return pots;

            auto ncalls = coords.num_calls();
            auto num_walkers = coords.num_walkers();

            // receive buffer -- needs to be the number of walkers total in the system,
            // so we take the number of walkers and multiply it into the number of calls we make
            PotValsManager pot_vals(1, 1);
            if ( world_rank() == 0) {
                pot_vals = PotValsManager(ncalls, num_walkers * world_size());
            }

            PyObject* gather = PyObject_GetAttrString(mpi_manager, "gather");
            GatherFunction gather_walkers = (GatherFunction) PyCapsule_GetPointer(gather, "Dumpi._GATHER_POTENTIALS");
            gather_walkers(
                    mpi_manager,
                    pots.data(),
                    num_walkers, // number of walkers fed in
                    pot_vals.data() // buffer to get the potential values back
            );
            Py_XDECREF(gather);

            return pot_vals;
        }

    } // PlzNumbers
};