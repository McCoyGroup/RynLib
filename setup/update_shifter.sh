#!/bin/bash
# from https://docs.nersc.gov/programming/shifter/how-to-use/

cd $(dirname $0)/..
bash ./update_docker.sh
docker tag rynimg registry.services.nersc.gov/b3m2a1/rynimg:latest
docker push registry.services.nersc.gov/b3m2a1/rynimg:latest