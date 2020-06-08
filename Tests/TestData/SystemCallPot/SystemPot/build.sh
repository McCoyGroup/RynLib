#! /bin/bash

# Very simple build file (can call make from here, but to start I wanted to have a build.sh file before making things more user friendly)
gcc -c -Wall -Werror -fPIC HarmonicOscillator.cpp
gcc -shared -o libHarmonicOscillator.so HarmonicOscillator.o #-install_name @rpath/libHarmonicOscillator.so
mv libHarmonicOscillator.so ../libHarmonicOscillator.so