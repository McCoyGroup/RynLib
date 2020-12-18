//
// Light-weight manager for walker configurations
//

#ifndef RYNLIB_COORDSMANAGER_HPP
#define RYNLIB_COORDSMANAGER_HPP

#include "RynTypes.hpp"

namespace rynlib {
    using namespace common;
    namespace PlzNumbers {
        class CoordsManager {

            RawWalkerBuffer walker_data;
            Names atoms;
            std::vector<size_t > shape;

        public:

            CoordsManager(
                    RawWalkerBuffer walkers,
                    Names& atom_names,
                    std::vector<size_t>& shape_vector,
                    bool new_obj = false
                    ) :
                    walker_data(walkers),
                    atoms(atom_names),
                    shape(shape_vector)
                    {};

            Coordinates get_walker(std::vector<size_t> which);
            Configurations get_walkers();
            FlatCoordinates get_flat_walker(std::vector<size_t> which);
            FlatConfigurations get_flat_walkers();
//            Configurations get_configurations();
            RawWalkerBuffer data() { return walker_data; }
            std::vector<size_t> get_shape() { return shape; }
            Names get_atoms() { return atoms; }
            size_t num_atoms() { return atoms.size(); }
            size_t num_calls() { return shape[1];}
            size_t num_walkers() { return shape[0];}
            size_t num_geoms() { return shape[0] * shape[1]; }

            PyObject* as_numpy_array();
            void cleanup() { free(walker_data); };
            // can't be called in the destructor
            // because python manages the initial set of structures that comes in...
            // I should move this all to a std::move/std::vector approach,
            // but at this point in time I just don't have the free time to clean it up

        };
    }
}


#endif //RYNLIB_COORDSMANAGER_HPP
