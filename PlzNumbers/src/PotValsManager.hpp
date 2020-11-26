#ifndef RYNLIB_POTVALSMANAGER_HPP
#define RYNLIB_POTVALSMANAGER_HPP

#include "RynTypes.hpp"

namespace rynlib {

    using namespace common;
    namespace PlzNumbers {
        class PotValsManager {
            PotentialVector pot_vals;
            size_t ncalls; // num_walkers uniquely determined by this

        public:

            PotValsManager(size_t num_calls, size_t num_walkers, Real_t fill=0.) : ncalls(num_calls) {
                auto tot_size = num_calls * num_walkers;
                pot_vals = PotentialVector(tot_size, fill);
            }

            PotValsManager(PotentialVector &pot_vector, size_t num_calls) :
                    pot_vals(pot_vector),
                    ncalls(num_calls) {}

            explicit PotValsManager(PotentialVector& pot_vector) : PotValsManager(pot_vector, 1) {}
            explicit PotValsManager() : PotValsManager(1, 1) {}
//            PotValsManager& operator(PotValsManager& pot) =


            size_t num_calls() { return ncalls; }
            size_t num_walkers() { return pot_vals.size() / ncalls; }
            std::vector<size_t> get_shape() {
                size_t buf[2] = {num_calls(), num_walkers()};
                std::vector<size_t> vec(buf, buf+2);
                return vec;
            }

            void assign(size_t n, size_t i, Real_t val);
            void assign(PotentialVector& new_vec) { pot_vals = new_vec; }
//            void assign(PotValsManager& new_manager) {
//                pot_vals = new_manager.vector();
//                ncalls = new_manager.num_calls();
//            }

            PotentialVector vector() { return pot_vals; }

            RawWalkerBuffer data() { return pot_vals.data(); }

        };
    }
}

#endif //RYNLIB_POTVALSMANAGER_HPP
