
#include "PotValsManager.hpp"

namespace rynlib {
    namespace PlzNumbers {
        void PotValsManager::assign(size_t n, size_t i, Real_t val) {
            auto nw = num_walkers();
            pot_vals[n*nw + i] = val;
        }
    }
}