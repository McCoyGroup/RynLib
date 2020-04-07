#!/bin/bash

cd $(dirname $0)

singularity build --fakeroot --docker-login entos SingularityEntos.def
singularity build --fakeroot rynlib Singularity.def

mv entos ../../entos
mv rynlib ../../rynlib