#!/bin/bash

RYNLIB_PATH=$(dirname $0); RYNLIB_PATH=$(dirname $RYNLIB_PATH); RYNLIB_PATH=$(dirname $RYNLIB_PATH); RYNLIB_PATH=$(dirname $RYNLIB_PATH)
export RYNLIB_PATH;

docker build -t rynlibcore -f $RYNLIB_PATH/setup/build/Docker/DockerfileCore $RYNLIB_PATH
docker build -t rynlib -f $RYNLIB_PATH/setup/build/Docker/Dockerfile $RYNLIB_PATH