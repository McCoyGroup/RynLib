#ifndef HARMONICOSCILLATOR_HPP

#include <vector>
#include <string>
#include <cmath>

extern double HarmonicOscillator(
    const std::vector< std::vector<double> > coords, const std::vector<std::string> atoms,
    float re, float k
    );

#define HARMONICOSCILLATOR_HPP

#endif //HARMONICOSCILLATOR_HPP
