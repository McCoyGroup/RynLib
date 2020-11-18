
#include "HarmonicOscillator.hpp"
#include <stdexcept>

double HarmonicOscillator(
    const std::vector< std::vector<double> > coords, const std::vector<std::string> atoms,
    float re, float k
    ) {
    double dist = 0;
    for (int i=0; i<3; i++) {
        dist += pow(coords[0][i] - coords[1][i], 2.0);
    }
    // here just for testing my error handlign
    if (dist < .0000001) {
        throw std::invalid_argument("Harmonic oscillator potential breaks down when bond length is 0");
    }

    dist = sqrt(dist) - re;
    double fc = (k/2.0);
    double pot = 1 * fc * pow(dist, 2.0);
    return pot;
}
