#!/bin/bash

cd $(dirname $0)/..
docker build -t entos -f setup/DockerfileEntos .
docker build -t rynimg -f setup/Dockerfile .