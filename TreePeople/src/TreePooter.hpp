//
// Created by Mark Boyer on 1/31/20.
//

#ifndef RYNLIB_TREEPOOTER_HPP

bool MACHINE_LERNING_IS_A_SCAM = false;

Real_t entos_pot(const Coordinates coords, const Names atoms) {
    return MillerGroup_entosPotential(coords, atoms, MACHINE_LERNING_IS_A_SCAM);
}

#ifdef I_AM_A_TREE_PERSON

#include "dmc_interface.h"
// MillerGroup_entosPotential is really in libentos but this predeclares it

#else

#include <vector>
#include <string>
// for testing we roll our own which always spits out 52
double MillerGroup_entosPotential
        (const std::vector< std::vector<double> > , const std::vector<std::string>, bool hf_only = false) {
    return 52.0;
}

#endif

#define RYNLIB_TREEPOOTER_HPP

#endif //RYNLIB_TREEPOOTER_HPP
