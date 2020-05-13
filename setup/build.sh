#!/bin/bash

RYNLIB_PATH=$(dirname $0); RYNLIB_PATH=$(dirname $RYNLIB_PATH)
export RYNLIB_PATH;

. $RYNLIB_PATH/setup/env.sh

push="$1"
if [[ "$push" = "--push" ]]; then
  build_type="$2"
else
  push=""
  build_type="$1"
fi

if [[ "$build_type" = "" ]]; then
  build_type="docker";
fi

if [[ "$build_type" = "docker" ]]; then
  docker build -t rynlibcore -f $RYNLIB_PATH/setup/build/Ubtuntu/RynlibCore $RYNLIB_PATH
  docker build -t $RYNLIB_IMAGE_NAME -f $RYNLIB_PATH/setup/build/Ubtuntu/RynlibBuild $RYNLIB_PATH
  if [[ "$push" == "--push" ]]; then
    docker tag $RYNLIB_IMAGE_NAME $RYNLIB_DOCKER_IMAGE
    docker push $RYNLIB_DOCKER_IMAGE
  fi
fi
if [[ "$build_type" = "ompi" ]]; then
  docker build -t rynlibcore -f $RYNLIB_PATH/setup/build/Ubtuntu/RynlibCore $RYNLIB_PATH
  docker build -t $RYNLIB_IMAGE_NAME-ompi -f $RYNLIB_PATH/setup/build/Ubtuntu-OpenMPI/RynlibBuild-OpenMPI $RYNLIB_PATH
  if [[ "$push" == "--push" ]]; then
    docker tag $RYNLIB_IMAGE_NAME $RYNLIB_DOCKER_IMAGE-ompi
    docker push $RYNLIB_DOCKER_IMAGE-ompi
  fi
fi
if [[ "$build_type" = "shifter" ]]; then
  docker build -t rynlibcore -f $RYNLIB_PATH/setup/build/Ubtuntu/RynlibCore $RYNLIB_PATH
  docker build -t $RYNLIB_IMAGE_NAME -f $RYNLIB_PATH/setup/build/Ubtuntu/RynlibBuild $RYNLIB_PATH
  if [[ "$push" == "--push" ]]; then
    docker tag $RYNLIB_IMAGE_NAME $RYNLIB_SHIFTER_IMAGE
    docker push $RYNLIB_SHIFTER_IMAGE
  fi
fi
if [[ "$build_type" = "singularity" ]]; then
  docker build -t rynlibcore-centos -f $RYNLIB_PATH/setup/build/CentOS/RynlibCoreCentOS $RYNLIB_PATH
  docker build -t $RYNLIB_IMAGE_NAME-centos -f $RYNLIB_PATH/setup/build/CentOS/RynlibBuildCentOS $RYNLIB_PATH
  if [[ "$push" == "--push" ]]; then
    docker tag $RYNLIB_IMAGE_NAME-centos $RYNLIB_DOCKER_IMAGE-centos
    docker push $RYNLIB_DOCKER_IMAGE-centos
  fi
fi