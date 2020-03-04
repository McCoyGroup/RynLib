from __future__ import print_function
from distutils.core import setup, Extension
import shutil, os, sys, subprocess

# Make it so src and all that stuff is available for use
curdir = os.getcwd()
lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_lib_dir = os.path.join(lib_dir, "libs")
src_dir=os.path.join(lib_dir, "src")
sysargv1 = sys.argv

# rebind sys.argv

lib_dirs = [ `LibDirs` ]
lib_dirs.append(lib_lib_dir)
libbies  = [ `LinkedLibs` ]
mroos = [ `LibMacros` ]

module = Extension(
    '`LibName`',
    sources = [ '`LibName`.cpp' ],
    library_dirs = lib_dirs,
    libraries = libbies,
    runtime_library_dirs = lib_dirs,
    define_macros = mroos
)

requires_make = `LibRequiresMake`
if requires_make:
    try:
        os.chdir(os.path.join(lib_lib_dir, '`LibName`'))
        subprocess.call(['bash', 'build.sh'])
    finally:
        os.chdir(curdir)

custom_build = `LibCustomBuild`
if not custom_build:
    try:
        sys.argv = ['build', 'build_ext', '--inplace']
        os.chdir(src_dir)

        setup(
            name = '`LibName`',
            version = '`LibVersion`',
            description = '`LibName` loves loops',
            ext_modules = [ module ],
            language = "c++"
        )
    finally:
        sys.argv = sysargv1
        os.chdir(curdir)


# Locate the library and copy it out (if it exists)
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
    # try:
    #     os.remove(target)
    # except:
    #     pass
    # os.rename(src, target)
    build_dir = os.path.join(lib_dir, "src", "build")
    if os.path.isdir:
        shutil.rmtree(build_dir)
else:
    failed = True