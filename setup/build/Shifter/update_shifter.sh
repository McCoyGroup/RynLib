#!/bin/bash
# from https://docs.nersc.gov/programming/shifter/how-to-use/

bash $(dirname $0)/update_docker.sh
docker tag rynimg:latest registry.services.nersc.gov/b3m2a1/rynimg:latest
docker push registry.services.nersc.gov/b3m2a1/rynimg:latest