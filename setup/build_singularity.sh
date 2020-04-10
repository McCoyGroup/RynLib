#!/bin/bash

cur=$PWD
cd $(dirname $0)

singularity build --fakeroot --docker-login entos SingularityEntos.def
singularity build --fakeroot rynlib Singularity.def

mv rynlib $cur/rynlib
cd $cur
# this is Hyak specific, for now...
mpi_lib="$1"
if [[ "$mpi_lib" == "" ]]; then
  module load gcc_8.2.1-ompi_3.1.4
  mpi_lib="/sw/openmpi/3.1.4-gcc-8.2.1/"
fi
rynlib="singularity run --bind .:/confg,$mpi_lib:/config/libs/mpi rynlib"
$rynlib config reload_dumpi