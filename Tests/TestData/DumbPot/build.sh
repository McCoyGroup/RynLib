#! /bin/bash

# Very simple build file (can call make from here, but to start I wanted to have a build.sh file before making things more user friendly)
gcc -c -Wall -Werror -fPIC DumbPot.cpp
gcc -shared -o libdumbpot.so DumbPot.o -install_name @rpath/libdumbpot.so
mv libdumbpot.so ../libdumbpot.so