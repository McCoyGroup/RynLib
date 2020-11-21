//
// Light-weight manager for walker configurations
//

#ifndef RYNLIB_CONFIGURATIONMANAGER_HPP
#define RYNLIB_CONFIGURATIONMANAGER_HPP

#include "RynTypes.hpp"

namespace rynlib {
    namespace PlzNumbers {
        class CoordsManager {

            RawWalkerBuffer walker_data;
            Names atoms;
            std::vector<int> shape;

        public:
            Coordinates get_walker(std::vector<size_t> which);
            Configurations get_walkers();
            FlatCoordinates get_flat_walker(std::vector<size_t> which);
            FlatConfigurations get_flat_walkers();
//            Configurations get_configurations();
            std::vector<int> get_shape() { return shape; }
            Names get_atoms() { return atoms; }
            size_t num_atoms() { return atoms.size(); }
            size_t num_walkers() { return shape[0] * shape[1]; }
        };
    }
}


#endif //RYNLIB_CONFIGURATIONMANAGER_HPP
