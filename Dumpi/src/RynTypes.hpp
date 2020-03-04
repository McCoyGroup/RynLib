//
// Created by Mark Boyer on 1/30/20.
//

#ifndef RYNLIB_RYNTYPES_HPP

#include <vector>
#include <string>

typedef double Real_t; // easy hook in case we wanted to use a different precision object or something in the future
typedef Real_t* RawWalkerBuffer;
typedef Real_t* RawPotentialBuffer;
typedef std::vector<Real_t> Point;
typedef Point PotentialVector;
typedef Point Weights;
typedef std::vector< Point > Coordinates;
typedef Coordinates PotentialArray;
typedef std::vector< Coordinates > Configurations;
typedef std::string Name;
typedef std::vector<std::string> Names;

typedef Real_t (*PotentialFunction)(const Coordinates , const Names);

#define RYNLIB_RYNTYPES_HPP

#endif //RYNLIB_RYNTYPES_HPP
