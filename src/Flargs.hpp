//
// Created by Mark Boyer on 1/30/20.
//

#ifndef RYNLIB_FLARGS_HPP

#include "RynLib.h"

std::string BAD_WALKERS_WHATCHA_GONNA_DO = "bad_walkers.txt";
bool MACHINE_LERNING_IS_A_SCAM = false;

Real_t entos_pot(const Coordinates coords, const Names atoms) {
    return MillerGroup_entosPotential(coords, atoms, MACHINE_LERNING_IS_A_SCAM);
}
PotentialFunction POOTY_PATOOTY = entos_pot;

#define RYNLIB_FLARGS_HPP

#endif //RYNLIB_FLARGS_HPP
