# Current NERSC Status

The basic call into Entos from python is working

The MPI calls are not _crashing_ but they're not _working_

The values don't seem to be coming back out of my MPI call when it calls into Entos in _mpiGetPot

For testing run:

```
rm RynLib/RynLib.so
python RynLib/test.py
```

It'll try to call the MPI stuff on a single water molecule, but we can move to more after that's working

If the compilation isn't working on NERSC, it *does* work on Victor's account where I made an env called b3m2a1

If you are getting build errors look inside `src/build.sh`

The parameters to customize look like:

```
homes_ext="r"
user="rjdiri"
venv="rjdiriEn"
conda="/global/homes/$homes_ext/$user/.conda/envs/$venv"
py_dir="$conda/include/python2.7"
```

where these just find the appropriate resources