//
// Created by Mark Boyer on 1/30/20.
//

#include "RynTypes.hpp"
#include "Flargs.hpp"
#include "PyAllUp.cpp"

void _printOutWalkerStuff( Coordinates walker_coords ) {
    if (!BAD_WALKERS_WHATCHA_GONNA_DO.empty()) {
        const char* fout = BAD_WALKERS_WHATCHA_GONNA_DO.c_str();
        FILE *err = fopen(fout, "a");
        fprintf(err, "This walker was bad: ( ");
        for (size_t i = 0; i < walker_coords.size(); i++) {
            fprintf(err, "(%f, %f, %f)", walker_coords[i][0], walker_coords[i][1], walker_coords[i][2]);
            if (i < walker_coords.size() - 1) {
                fprintf(err, ", ");
            }
        }
        fprintf(err, " )\n");
        fclose(err);
    } else {
        printf("This walker was bad: ( ");
        for ( size_t i = 0; i < walker_coords.size(); i++) {
            printf("(%f, %f, %f)", walker_coords[i][0], walker_coords[i][1], walker_coords[i][2]);
            if ( i < walker_coords.size()-1 ) {
                printf(", ");
            }
        }
        printf(" )\n");
    }
}


// Basic method for computing a potential via the global potential bound in POOTY_PATOOTY
double _doopAPot(const Coordinates &walker_coords, const Names &atoms) {
    double pot;

    try {
        pot = POOTY_PATOOTY(walker_coords, atoms);
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        _printOutWalkerStuff(walker_coords);
        pot = 1.0e9;
    }

    return pot;
};



inline int ind2d(int i, int j, int n, int m) {
    return m * i + j;
}

// here I ignore `n` because... well I originally wrote it like that
inline int int3d(int i, int j, int k, int m, int l) {
    return (m*l) * i + (l*j) + k;
}

Coordinates _getWalkerCoords(const double* raw_data, int i, Py_ssize_t num_atoms) {
    Coordinates walker_coords(num_atoms, Point(3));
    for (int j = 0; j<num_atoms; j++) {
        for (int k = 0; k<3; k++) {
            walker_coords[j][k] = raw_data[int3d(i, j, k, num_atoms, 3)];
        }
    };
    return walker_coords;
}

inline int int4d(int i, int j, int k, int a, int n, int m, int l, int o) {
    return (m*l*o) * i + (l*o*j) + o*k + a;
}

// pulls data for the ith walker in the nth call
// since we start out with data that looks like (ncalls, nwalkers, ...)
Coordinates _getWalkerCoords2(const double* raw_data, int n, int i, int ncalls, int num_walkers, Py_ssize_t num_atoms) {
    Coordinates walker_coords(num_atoms, Point(3));
    for (int j = 0; j<num_atoms; j++) {
        for (int k = 0; k<3; k++) {
            walker_coords[j][k] = raw_data[int4d(n, i, j, k, ncalls, num_walkers, num_atoms, 3)];
        }
    };
    return walker_coords;
}
