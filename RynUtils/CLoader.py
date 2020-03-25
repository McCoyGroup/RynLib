import shutil, os, sys, subprocess, importlib, platform
from distutils.core import setup, Extension

__all__ = [
    "CLoader"
]

class CLoader:
    """
    A general loader for C++ extensions to python, based off of the kind of thing that I have had to do multiple times
    """

    def __init__(self,
                 lib_name,
                 lib_dir,
                 src_ext = 'src',
                 description = "An extension module",
                 verion = "1.0.0",
                 include_dirs = None,
                 linked_libs = None,
                 macros = None,
                 extra_link_args=None,
                 extra_compile_args=None,
                 extra_objects=None,
                 source_files = None,
                 build_script = None,
                 requires_make = False,
                 out_dir = None,
                 cleanup_build = True
                 ):
        self.lib_name = lib_name
        self.lib_dir = lib_dir
        self.lib_description = description
        self.lib_version = verion

        self.include_dirs = () if include_dirs is None else tuple(include_dirs)
        self.linked_libs = () if linked_libs is None else tuple(linked_libs)
        self.extra_link_args = () if extra_link_args is None else tuple(extra_link_args)
        self.extra_compile_args = () if extra_compile_args is None else tuple(extra_compile_args)
        self.extra_objects = () if extra_objects is None else tuple(extra_objects)
        self.src_ext = src_ext
        self.macros = () if macros is None else tuple(macros)
        self.source_files = (lib_name+'.cpp',) if source_files is None else source_files
        self.build_script = build_script
        self.requires_make = requires_make

        self.out_dir = out_dir
        self.cleanup_build = cleanup_build

        self._lib = None

    def load(self):
        if self._lib is None:
            ext = self.find_extension()
            if ext is None:
                ext = self.compile_extension()
            if ext is None:
                raise ImportError("Couldn't load/compile extension {} at {}".format(
                    self.lib_name,
                    self.lib_dir
                ))
            try:
                sys.path.insert(0, os.path.dirname(ext))
                module = os.path.splitext(os.path.basename(ext))[0]
                self._lib = importlib.import_module(module, self.lib_name+"Lib")
            finally:
                sys.path.pop(0)

        return self._lib

    def find_extension(self):
        """
        Tries to find the extension in the top-level directory

        :return:
        :rtype:
        """

        return self.locate_lib(self.lib_dir)[0]

    def compile_extension(self):
        """
        Compiles and loads a C++ extension

        :return:
        :rtype:
        """

        self.make_required_libs()
        self.build_lib()
        ext = self.cleanup()
        return ext

    @property
    def src_dir(self):
        return os.path.join(self.lib_dir, self.src_ext)
    @property
    def lib_lib_dir(self):
        return os.path.join(self.lib_dir, "libs")

    def get_extension(self):
        """
        Gets the Extension module to be compiled

        :return:
        :rtype:
        """
        lib_lib_dir = self.lib_lib_dir

        lib_dirs = self.include_dirs + (lib_lib_dir,)
        libbies = self.linked_libs
        mroos = self.macros
        sources = self.source_files

        extra_link_args = list(self.extra_link_args)
        extra_compile_args = list(self.extra_compile_args)
        if platform.system() == 'Darwin':
            extra_link_args.append('-Wl,-rpath,' + ":".join(lib_dirs))

        module = Extension(
            self.lib_name,
            sources=list(sources),
            library_dirs=list(lib_dirs),
            runtime_library_dirs=list(lib_dirs),
            libraries=list(libbies),
            define_macros=list(mroos),
            extra_objects=list(self.extra_objects),
            extra_link_args=extra_link_args,
            extra_compile_args=extra_compile_args,
            language="c++"
        )

        return module

    def custom_make(self, make_file, make_dir):
        """
        A way to call a custom make file either for building the helper lib or for building the proper lib

        :param make_file:
        :type make_file:
        :param make_dir:
        :type make_dir:
        :return:
        :rtype:
        """

        curdir = os.getcwd()

        if isinstance(make_file, str) and os.path.isfile(make_file):
            make_dir = os.path.dirname(make_file)
            make_file = os.path.basename(make_file)
            if os.path.splitext(make_file)[1] == ".sh":
                make_cmd = ["bash", make_file]
            else:
                make_cmd = ["make", "-f", make_file]
        else:

            if os.path.exists(os.path.join(make_dir, "Makefile")):
                make_cmd = ["make"]
            elif os.path.exists(os.path.join(make_dir, "build.sh")):
                make_cmd = ["bash", "build.sh"]
            else:
                raise IOError(
                    "Can't figure out which file in {} should be the makefile. Expected either Makefile or build.sh".format(
                        make_dir
                    )
                )

        try:
            os.chdir(make_dir)
            subprocess.call(make_cmd)
        finally:
            os.chdir(curdir)

    def make_required_libs(self):
        """
        Makes any libs required by the current one

        :return:
        :rtype:
        """
        if self.requires_make:
            self.custom_make(self.requires_make, os.path.join(self.lib_lib_dir, self.lib_name))

    def build_lib(self):

        curdir = os.getcwd()

        src_dir = self.src_dir
        module = self.get_extension()

        sysargv1 = sys.argv
        custom_build = self.build_script
        if custom_build:
            self.custom_make(custom_build, src_dir)
        else:
            try:
                sys.argv = ['build', 'build_ext', '--inplace']
                os.chdir(src_dir)

                setup(
                    name=self.lib_description,
                    version=self.lib_version,
                    description=self.lib_description,
                    ext_modules=[module]
                )
            finally:
                sys.argv = sysargv1
                os.chdir(curdir)

    def locate_lib(self, root = None):
        """
        Tries to locate the build library file (if it exists)

        :return:
        :rtype:
        """

        libname = self.lib_name
        lib_dir = self.lib_dir
        if root is None:
            root = os.path.join(lib_dir, "src")
        target = libname
        built = None
        ext = ""

        for f in os.listdir(root):
            if f.startswith(libname) and f.endswith(".so"):
                ext = ".so"
                built = os.path.join(root, f)
                target += ext
                break
            elif f.startswith(libname) and f.endswith(".pyd"):
                ext = ".pyd"
                built = os.path.join(root, f)
                target += ext
                break

        if built is None:
            target = None

        return built, target, ext

    def cleanup(self):
        # Locate the library and copy it out (if it exists)

        built, target, ext = self.locate_lib()

        if built is not None:
            target_dir = self.out_dir
            if target_dir is None:
                target_dir = self.lib_dir
            target = os.path.join(target_dir, target)
            try:
                os.remove(target)
            except:
                pass
            os.rename(built, target)

            if self.cleanup_build:
                build_dir = os.path.join(self.src_dir, "build")
                if os.path.isdir:
                    shutil.rmtree(build_dir)

        return target