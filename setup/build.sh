#!/bin/bash

RYNLIB_PATH=$(dirname $0); RYNLIB_PATH=$(dirname $RYNLIB_PATH)
export RYNLIB_PATH;

. $RYNLIB_PATH/setup/env.sh

build_type="$1"

if [[ "$build_type" = "" ]]; then
  build_type="docker";
fi

if [[ "$build_type" = "docker" ]]; then
  docker build -t rynlibcore -f $RYNLIB_PATH/setup/build/Docker/RynlibCore $RYNLIB_PATH
  docker build -t $RYNLIB_IMAGE_NAME -f $RYNLIB_PATH/setup/build/Docker/RynlibBuild $RYNLIB_PATH
  docker tag $RYNLIB_IMAGE_NAME $RYNLIB_DOCKER_IMAGE
  docker push $RYNLIB_DOCKER_IMAGE
fi
if [[ "$build_type" = "shifter" ]]; then
  docker build -t rynlibcore -f $RYNLIB_PATH/setup/build/Docker/RynlibCore $RYNLIB_PATH
  docker build -t $RYNLIB_IMAGE_NAME -f $RYNLIB_PATH/setup/build/Docker/RynlibBuild $RYNLIB_PATH
  docker tag $RYNLIB_IMAGE_NAME $RYNLIB_SHIFTER_IMAGE
  docker push $RYNLIB_SHIFTER_IMAGE
fi
if [[ "$build_type" = "singularity" ]]; then
  docker build -t rynlibcore-centos -f $RYNLIB_PATH/setup/build/Docker/RynlibCoreCentOS $RYNLIB_PATH
  docker build -t $RYNLIB_IMAGE_NAME-centos -f $RYNLIB_PATH/setup/build/Docker/RynlibBuildCentOS $RYNLIB_PATH
  docker tag $RYNLIB_IMAGE_NAME-centos $RYNLIB_DOCKER_IMAGE-centos
  docker push $RYNLIB_DOCKER_IMAGE-centos
fi