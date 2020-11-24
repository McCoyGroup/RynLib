//
// Manage coordinate defs
//

#include "CoordsManager.hpp"
#include "PyAllUp.hpp"
#include <stdexcept>

namespace rynlib {
    namespace PlzNumbers {

        // dumb little function to get indices
        inline int ind2d(int i, int j, int n, int m) {
            return m * i + j;
        }

        // here I ignore `n` because... well I originally wrote it like that
        inline int int3d(int i, int j, int k, int m, int l) {
            return (m * l) * i + (l * j) + k;
        }

        inline int int4d(int i, int j, int k, int a, int n, int m, int l, int o) {
            return (m * l * o) * i + (l * o * j) + o * k + a;
        }

        void _fillWalkerCoords(const double *raw_data, int i, Py_ssize_t num_atoms,
                               Coordinates &walker_coords) {
            for (int j = 0; j < num_atoms; j++) {
                for (int k = 0; k < 3; k++) {
                    walker_coords[j][k] = raw_data[int3d(i, j, k, num_atoms, 3)];
                }
            };
        }

        Coordinates _getWalkerCoords(const double *raw_data, int i, Py_ssize_t num_atoms) {
            Coordinates walker_coords(num_atoms, Point(3));
            _fillWalkerCoords(raw_data, i, num_atoms, walker_coords);
            return walker_coords;
        }


        FlatCoordinates _getWalkerFlatCoords(const double *raw_data, int i, Py_ssize_t num_atoms) {
            FlatCoordinates walker_coords(num_atoms * 3);
            for (int j = 0; j < num_atoms; j++) {
                for (int k = 0; k < 3; k++) {
                    double crd = raw_data[int3d(i, j, k, num_atoms, 3)];
//            printf("...%f\n", crd);
                    walker_coords[ind2d(j, k, num_atoms, 3)] = crd;
                }
            };
            return walker_coords;
        }

        // pulls data for the ith walker in the nth call
        // since we start out with data that looks like (ncalls, nwalkers, ...)
        void _fillWalkerCoords2(const double *raw_data, int n, int i, int ncalls, int num_walkers,
                                Py_ssize_t num_atoms,
                                Coordinates &walker_coords
        ) {
            for (int j = 0; j < num_atoms; j++) {
                for (int k = 0; k < 3; k++) {
                    walker_coords[j][k] = raw_data[int4d(n, i, j, k, ncalls, num_walkers, num_atoms, 3)];
                }
            };
        }

        Coordinates
        _getWalkerCoords2(const double *raw_data, int n, int i, int ncalls, int num_walkers, Py_ssize_t num_atoms) {
            Coordinates walker_coords(num_atoms, Point(3));
            _fillWalkerCoords2(raw_data, n, i, ncalls, num_walkers, num_atoms, walker_coords);
            return walker_coords;
        }

        FlatCoordinates
        _getWalkerFlatCoords2(const double *raw_data, int n, int i, int ncalls, int num_walkers, Py_ssize_t num_atoms) {
            FlatCoordinates walker_coords(num_atoms * 3);
            for (int j = 0; j < num_atoms; j++) {
                for (int k = 0; k < 3; k++) {
                    walker_coords[ind2d(j, k, num_atoms, 3)] = raw_data[
                            int4d(n, i, j, k, ncalls, num_walkers, num_atoms, 3)
                    ];
                }
            };
            return walker_coords;
        }

        Coordinates CoordsManager::get_walker(std::vector<size_t> which) {
            Coordinates walker_coords;
            size_t num_crds = which.size();
//            auto ncalls = num_calls();
//            auto nwalks = num_walkers();
            switch (num_crds) {
                case 1:
                    walker_coords = _getWalkerCoords(
                            walker_data,
                            which[0],
                            num_atoms()
                    );
                    break;
                case 2:
                    walker_coords = _getWalkerCoords2(
                            walker_data,
                            which[0],
                            which[1],
                            num_calls(),
                            num_walkers(),
                            num_atoms()
                    );
                    break;
                default:
                    throw std::logic_error("bad number of indices to CoordsManager::get_walker");
            }
            return walker_coords;
        }

        FlatCoordinates CoordsManager::get_flat_walker(std::vector<size_t> which) {
            FlatCoordinates walker_coords;
            size_t num_crds = which.size();
            auto ncalls = num_calls();
            auto nwalks = num_walkers();
            switch (num_crds) {
                case 1:
                    walker_coords = _getWalkerFlatCoords(
                            walker_data,
                            which[0],
                            num_atoms()
                    );
                    break;
                case 2:
                    walker_coords = _getWalkerFlatCoords2(
                            walker_data,
                            which[0],
                            which[1],
                            ncalls,
                            nwalks,
                            num_atoms()
                    );
                    break;
                default:
                    throw std::runtime_error("bad number of indices to CoordsManager::get_flat_walker");
            }
            return walker_coords;
        }

        Configurations CoordsManager::get_walkers() {
            // construct array and fill...not great but not a total disaster
            // probably could be done more elegantly with some direct initialization or
            // something but I'm dumb and don't know C++ that well...
            Configurations walkers(num_geoms(),
                                   Coordinates(num_atoms(), Point(3, 0.))
            );
            auto ncalls = num_calls();
            auto nwalks = num_walkers();
            for (size_t n = 0; n < ncalls; n++) {
                for (size_t i = 0; i < nwalks; i++) {
                    _fillWalkerCoords2(
                            walker_data, n, i,
                            ncalls, nwalks, num_atoms(),
                            walkers[n]
                    );
                }
            }
            return walkers;
        };

        FlatConfigurations CoordsManager::get_flat_walkers() { // dope
            return walker_data;
        };

        PyObject* CoordsManager::as_numpy_array() {
            // ncalls and nconfigs get transposed when going back to NumPy; doesn't really matter in general

            size_t wat_shape[4] {shape[1], shape[0], num_atoms(), 3}; // C++ didn't like my vector initializer ?
//            printf("...? 222\n");
            std::vector<size_t > np_shape(wat_shape, wat_shape+4);
//            printf("...? 333\n");
            return python::numpy_from_data<Real_t >(
                    walker_data,
                    np_shape
                    );

        }

    }// PlzNumbers
}