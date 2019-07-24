# used for building python on NERSC since we've run into issues there

cd $(dirname "$0")

compy="CC"
linky="CC"

ryn_lib_dir=$(dirname ${PWD})
homes_ext="b"
user="b3m2a1"
venv="b3m2a1-python2.7"
conda="/usr/common/software/python/2.7-anaconda-4.4"
if [ ! -d "$conda" ]
    then conda="/global/homes/$homes_ext/$user/.conda/envs/$venv"
fi
py_dir="$conda/include/python2.7"

mkdir "build"

#compile source
comp_com="CC -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -DIM_A_REAL_BOY -I$py_dir -c RynLib.cpp -o build/RynLib.o"
echo $comp_com
$comp_com

#link entos and stuff
lib_dir="$ryn_lib_dir/lib"
link_com="CC -shared build/RynLib.o -L$lib_dir -L$conda/lib -Wl,-R$lib_dir -lentos -lecpint -lintception -lpython2.7 -o RynLib.so"
echo $link_com
$link_com
