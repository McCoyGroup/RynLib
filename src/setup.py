from distutils.core import setup, Extension
import shutil, os, sys

curdir = os.getcwd()
lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(lib_dir, "src"))
sysargv1 = sys.argv

sys.argv = ['build', 'build_ext', '--inplace']
module = Extension(
    'RynLib',
    sources = [ 'RynLib.cpp' ],
    library_dirs= [ os.path.join(lib_dir, "lib") ],
    libraries = [ "entos" ]
)

setup (name = 'RynLib',
       version = '1.0',
       description = 'Ryna Dorisii loves loops',
       ext_modules = [module],
       language="c++"
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