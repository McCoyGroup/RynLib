
#ifndef RYNLIB_RYNTYPES_HPP

#include <vector>
#include <string>
#include "Python.h"

typedef double Real_t; // easy hook in case we wanted to use a different precision object or something in the future
typedef Real_t* RawWalkerBuffer;
typedef Real_t* RawPotentialBuffer;
typedef std::vector<Real_t> Point;
typedef Point PotentialVector;
typedef Point Weights;
typedef std::vector< Point > Coordinates;
typedef Point FlatCoordinates;
typedef Coordinates PotentialArray;
typedef std::vector< Coordinates > Configurations;
typedef std::string Name;
typedef std::vector<std::string> Names;

typedef std::vector<bool> ExtraBools;
typedef std::vector<int> ExtraInts;
typedef std::vector<Real_t> ExtraFloats;

typedef Real_t (*PotentialFunction)(
        const Coordinates,
        const Names,
        const ExtraBools,
        const ExtraInts,
        const ExtraFloats
);
typedef Real_t (*FlatPotentialFunction)(
        const FlatCoordinates,
        const Names,
        const ExtraBools,
        const ExtraInts,
        const ExtraFloats
);

typedef int (*ScatterFunction)(PyObject*, RawWalkerBuffer, int, int, RawWalkerBuffer);
typedef int (*GatherFunction)(PyObject*, RawPotentialBuffer, int, RawPotentialBuffer);

#define RYNLIB_RYNTYPES_HPP

#endif //RYNLIB_RYNTYPES_HPP
