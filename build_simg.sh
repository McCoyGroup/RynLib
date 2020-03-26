#!/bin/bash

cd $(dirname $0)
docker build -t entos -f DockerfileEntos .
docker build -t ryn_app .