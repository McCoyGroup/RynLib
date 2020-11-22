
#ifndef SIMPLEFFI_FFIPARAMETERS_HPP
#define SIMPLEFFI_FFIPARAMETERS_HPP

#include <string>
#include <vector>
#include <exception>

namespace simpleffi {
        class FFIParameters {
            std::vector<void* > params; // type info resolved through <get_arg>
            std::vector<std::string> keys;
        public:
            FFIParameters(
                    std::vector<std::string> key_vec,
                    std::vector<void* > param_vec
                    ) : keys(key_vec), params(param_vec) {};

            size_t param_index(std::string key) {
                size_t idx = 0;
                for (idx=0; idx < keys.size(); idx++) {
                    auto k = keys[idx];
                    if (k == key) break;
                    idx += 1;
                }

                if (idx == keys.size()) throw std::runtime_error("index missing...");

                return idx;

            }

            template <typename T>
            T get_arg(std::string key) {
                return params[param_index(key)];
            }

            template <typename T>
            void set_arg(std::string key, T val) {
                params[param_index(key)] = val;
            }


        };
}

#endif //SIMPLEFFI_FFIPARAMETERS_HPP
