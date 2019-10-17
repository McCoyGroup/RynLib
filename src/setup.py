from __future__ import print_function
from distutils.core import setup, Extension
import shutil, os, sys, getpass, platform

curdir = os.getcwd()
lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir=os.path.join(lib_dir, "src")
os.chdir(src_dir)
sysargv1 = sys.argv

sys.argv = ['build', 'build_ext', '--inplace']

lib_dirs = []
libbies = []
mrooos = []

node_name = platform.node()

on_nersc = (
    # we'll support Mox too I guess...
    node_name.startswith("cori") or node_name.startswith("nid") # a flag to set when building on nersc
)

sadboydebug=False

if not on_nersc:
    print("you're not a on NERSC, I'm not using entos")
else:
    if sadboydebug:
        mrooos.append(("SADBOYDEBUG", None))
    else:
        lib_dirs.append(os.path.join(lib_dir, "lib"))
        libbies.extend(("entos", "ecpint", "intception"))
        mrooos.append(("IM_A_REAL_BOY", None))

module = Extension(
    'RynLib',
    sources = [ 'RynLib.cpp' ],
    library_dirs = lib_dirs,
    libraries = libbies,
    runtime_library_dirs = lib_dirs,
    define_macros = mrooos
)

if on_nersc:
    print("Using build.sh to get around NERSC issues")
    os.system(os.path.join(src_dir, "build.sh"))
else:
    setup(
       name = 'RynLib',
       version = '1.0',
       description = 'Ryna Dorisii loves loops',
       ext_modules = [module],
       language = "c++"
       )

os.chdir(curdir)

ext = ""
libname="RynLib"
target = os.path.join(lib_dir, libname)
src = None

for f in os.listdir(os.path.join(lib_dir, "src")):
    if f.startswith(libname) and f.endswith(".so"):
        ext = ".so"
        src = os.path.join(lib_dir, "src", f)
        target += ext
        break
    elif f.startswith(libname) and f.endswith(".pyd"):
        ext = ".pyd"
        src = os.path.join(lib_dir, "src", f)
        target += ext
        break

if src is not None:
    failed = False
    try:
        os.remove(target)
    except:
        pass
    os.rename(src, target)
    shutil.rmtree(os.path.join(lib_dir, "src", "build"))
else:
    failed = True