# used for building python on NERSC since we've run into issues there

cd $(dirname "$0")

ryn_lib_dir=$(dirname ${PWD})
homes_ext="b"
user="b3m2a1"
venv="b3m2a1-python2.7"
conda="/usr/common/software/python/2.7-anaconda-2019.07"
if [ ! -d "$conda" ]
    then conda="/global/homes/$homes_ext/$user/.conda/envs/$venv"
fi
py_dir="$conda/include/python2.7"


#compy="CC"
#compy="g++"
compy=mpic++
#linky="CC"
#linky="g++"
linky=mpic++

mkdir "build"

#compile source
#comp_flags="-fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes""
#comp_flags="-pthread -fPIC -B $conda/compiler_compat -Wl,--sysroot=/"
#comp_flags="-pthread -fno-strict-aliasing -fmessage-length=0 -grecord-gcc-switches -O2 -Wall -D_FORTIFY_SOURCE=2 -fstack-protector-strong -funwind-tables -fasynchronous-unwind-tables -fstack-clash-protection -g -DNDEBUG -fmessage-length=0 -grecord-gcc-switches -O2 -Wall -D_FORTIFY_SOURCE=2 -fstack-protector-strong -funwind-tables -fasynchronous-unwind-tables -fstack-clash-protection -g -DOPENSSL_LOAD_CONF -fwrapv"
comp_flags=""
#build_flags="-DSADBOYDEBUG"
build_flags="-DIM_A_REAL_BOY"
#build_flags=""
include_dir=$py_dir
#include_dir="/usr/include/python2.7"
comp_com="$compy $comp_flags $build_flags -fPIC -I$include_dir -c RynLib.cpp -o build/RynLib.o"
echo $comp_com
$comp_com

#link entos and stuff
lib_dir="$ryn_lib_dir/lib"
#link_flags="-shared -Wl,-R$lib_dir"
#link_flags="-pthread -shared -B $conda/compiler_compat -Wl,-rpath=$conda/lib -Wl,--no-as-needed -Wl,--sysroot=/"
link_flags="-pthread -shared -Wl,-R$lib_dir"
#link_flags="-pthread -shared"
#link_libs="-L/usr/lib64"
#link_libs="-L$conda/lib"
#link_libs="-L$lib_dir -L$conda/lib -lentos -lecpint -lintception"
link_libs="-L$lib_dir -L/usr/lib64 -lentos -lecpint -lintception"
link_com="$linky $link_flags build/RynLib.o $link_libs -lpython2.7 -o RynLib.so"
echo $link_com
$link_com

