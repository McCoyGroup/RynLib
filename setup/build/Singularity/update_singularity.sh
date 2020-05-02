#!/bin/bash

cur=$PWD
cd $(dirname $0)

singularity build --fakeroot rynlib SingularityUpdate.def

cp rynlib $cur/rynlib
cd $cur