#!/bin/bash

cd $(dirname $0)
docker build -t entos -f DockerfileEntos .
docker build --no-cache -t rynimg -f Dockerfile .