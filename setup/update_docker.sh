#!/bin/bash

cd $(dirname $0)/..
docker build --no-cache -t rynimg -f setup/DockerfileUpdate .