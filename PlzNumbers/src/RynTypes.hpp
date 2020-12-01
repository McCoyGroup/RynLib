
#ifndef RYNLIB_RYNTYPES_HPP

#include <vector>
#include <string>
#include "Python.h"

namespace rynlib::common {

        // annoying
        bool debug_print();
        void set_debug_print(bool db);

        typedef double Real_t; // easy hook in case we wanted to use a different precision object or something in the future
        typedef Real_t *RawWalkerBuffer;
        typedef Real_t *RawPotentialBuffer;
        typedef std::vector<Real_t> Point;
        typedef Point PotentialVector;
        typedef Point Weights;
        typedef std::vector<Point> Coordinates;
        typedef Point FlatCoordinates;
        typedef Coordinates PotentialArray;
        typedef std::vector<Coordinates> Configurations;
        typedef RawPotentialBuffer FlatConfigurations;
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

        typedef PotentialVector (*VectorizedPotentialFunction)(
                const Configurations,
                const Names,
                const ExtraBools,
                const ExtraInts,
                const ExtraFloats
        );

        typedef RawPotentialBuffer (*VectorizedFlatPotentialFunction)(
                const FlatConfigurations,
                const Names,
                const ExtraBools,
                const ExtraInts,
                const ExtraFloats
        );

}
#define RYNLIB_RYNTYPES_HPP

#endif //RYNLIB_RYNTYPES_HPP
