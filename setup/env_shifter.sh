

function pull_rynlib() { shifterimg pull registry.services.nersc.gov/b3m2a1/rynimg:latest };
rynlib="shifter --volume=$PWD:/config --image=registry.services.nersc.gov/b3m2a1/rynimg:latest python3.7 /home/RynLib/CLI.py"
