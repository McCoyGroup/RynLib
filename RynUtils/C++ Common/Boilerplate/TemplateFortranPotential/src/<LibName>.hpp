#ifndef RYNLIB_<LibName>_POT_HPP

#include "Python.h"
#include "RynTypes.hpp"

/*
 * Find and replace <LibName> with the name of the library
 * Find and replace <LibFunction> with the name of the function
 * We also need to declare the call signature for our Fortran function here, even though it'll just link in at
 * runtime. We do that by replacing <CallSignature>.
 * Usually we'll want a RawWalkerBuffer (array of coordinate doubles) and the energy needs to come out somewhere as a
 * int* (pointer to an integer in memory).
 */
Real_t <LibName>_<LibFunction>(
    const FlatCoordinates,
    const Names,
    const ExtraBools,
    const ExtraInts,
    const ExtraFloats
    );
extern "C" { // this is to keep C++ from mangling the function name
    void <LibFunction>(<CallSignature>)
}

#define RYNLIB_<LibName>_POT_HPP

#endif //RYNLIB_<LibName>_POT_HPP
