#!/bin/bash

cd $(dirname $0)/..
docker build -t entos -f setup/DockerfileEntos .
docker build --no-cache -t rynimg -f setup/Dockerfile .

rynlib="docker run --rm --mount source=simdata,target=/config -it rynimg"
$rynlib config reload_dumpi