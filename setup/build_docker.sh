#!/bin/bash

cd $(dirname $0)
docker build --no-cache -t entos -f DockerfileEntos .
docker build --no-cache -t rynimg -f Dockerfile .

rynlib="docker run --rm --mount source=simdata,target=/config -it rynimg"
$rynlib config reload_dumpi