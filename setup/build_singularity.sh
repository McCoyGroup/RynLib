#!/bin/bash

cur=$PWD
cd $(dirname $0)

singularity build --fakeroot --docker-login entos SingularityEntos.def
singularity build --fakeroot rynlib Singularity.def

mv rynlib $cur/rynlib
cd $cur
./rynlib config reload_dumpi