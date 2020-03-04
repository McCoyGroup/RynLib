#! /bin/bash

# Very simple build file (can call make from here, but to start I wanted to have a build.sh file before making things more user friendly)
gcc -c -Wall -Werror -fpic DumbPot.cpp
gcc -shared -o libDumbPot.so DumbPot.o
mv libDumbPot.so ../libDumbPot.so