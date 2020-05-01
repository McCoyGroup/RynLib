

rynlib="docker run --rm --mount source=simdata,target=/config -it rynimg"
function ryndata() { echo "docker run --rm --mount source=simdata,target=/config -it -v $1:rw rynimg"; }