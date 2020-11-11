
#ifndef PLZNUMBERS_POTATORS_HPP

#include "RynTypes.hpp"
#include "Python.h"
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void _printOutWalkerStuff( Coordinates walker_coords );

double _doopAPot(
        Coordinates &walker_coords,
        Names &atoms,
        PotentialFunction pot_func,
        std::string &bad_walkers_file,
        double err_val,
        bool debug_print,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats,
        int retries = 3
        );
double _doopAPot(
        FlatCoordinates &walker_coords,
        Names &atoms,
        FlatPotentialFunction pot_func,
        std::string &bad_walkers_file,
        double err_val,
        bool debug_print,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats,
        int retries = 3
);

inline int ind2d(int i, int j, int n, int m);
inline int int3d(int i, int j, int k, int m, int l);
FlatCoordinates _getWalkerFlatCoords(const double* raw_data, int i, Py_ssize_t num_atoms);
Coordinates _getWalkerCoords(const double* raw_data, int i, Py_ssize_t num_atoms);

inline int int4d(int i, int j, int k, int a, int n, int m, int l, int o);
Coordinates _getWalkerCoords2(const double* raw_data, int n, int i, int ncalls, int num_walkers, Py_ssize_t num_atoms);

PotentialArray _mpiGetPot(
        PyObject* manager,
        PotentialFunction pot,
        FlatPotentialFunction flat_pot,
        RawWalkerBuffer raw_data,
        Names &atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        PyObject* bad_walkers_file,
        double err_val,
        bool debug_print,
        int retries,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats,
        bool use_openMP,
        bool use_TBB
        );

PotentialArray _noMPIGetPot(
        PotentialFunction pot,
        FlatPotentialFunction flat_pot,
        double* raw_data,
        Names &atoms,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        PyObject* bad_walkers_file,
        double err_val,
        bool debug_print,
        int retries,
        ExtraBools &extra_bools,
        ExtraInts &extra_ints,
        ExtraFloats &extra_floats,
        bool use_openMP,
        bool use_TBB
        );

PyObject* _mpiGetPyPot(
        PyObject* manager,
        PyObject* pot_func,
        RawWalkerBuffer raw_data,
        PyObject* atoms,
        PyObject* extra,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms
);

#define PLZNUMBERS_POTATORS_HPP

#endif //PLZNUMBERS_POTATORS_HPP
