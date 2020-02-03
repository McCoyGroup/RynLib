from __future__ import print_function
from distutils.core import setup, Extension
import shutil, os, sys

curdir = os.getcwd()
lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir=os.path.join(lib_dir, "src")
os.chdir(src_dir)
sysargv1 = sys.argv

sys.argv = ['build', 'build_ext', '--inplace']

lib_dirs = []
libbies  = []

module = Extension(
    '`LibName`',
    sources = [ '`LibName`.cpp' ],
    library_dirs = lib_dirs,
    libraries = libbies,
    runtime_library_dirs = lib_dirs,
    define_macros = []
)

setup(
    name = '`LibName`',
    version = '1.0',
    description = '`LibName` loves loops',
    ext_modules = [module],
    language = "c++"
)

os.chdir(curdir)

ext = ""
libname = "`LibName`"
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