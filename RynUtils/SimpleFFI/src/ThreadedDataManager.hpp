//
// Light-weight manager for walker configurations
//

#ifndef SIMPLEFFI_THREADEDDATAMANAGER_HPP
#define SIMPLEFFI_THREADEDDATAMANAGER_HPP

#include "RynTypes.hpp"

namespace simpleffi {
    using namespace common;

    template <typename T>
    class ThreadedDataManager {

        T* data_buf;
        // we don't recompose into a std vec until later
        // & then only if we need it
        std::vector <size_t> shape_vec;


    public:

        ThreadedDataManager(
                T* dat_buffer,
                std::vector<size_t> shape_vector
        ) :
                data_buf(dat_buffer),
                shape_vec(shape_vector) {};

        T get_structure(std::vector <size_t> which);
        std::vector<T>  get_structures();
        T* data() { return data_buf; }
        std::vector <size_t> shape { return shape; }

        void assign(size_t n, size_t i, Real_t val);
        void assign(PotentialVector& new_vec) { pot_vals = new_vec; }
        void assign(PotValsManager& new_manager) {
            pot_vals = new_manager.vector();
            ncalls = new_manager.num_calls();
        }

    };
}


#endif //SIMPLEFFI_THREADEDDATAMANAGER_HPP
