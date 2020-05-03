import importlib.abc, os, importlib.util, sys

__all__ = [
    "ModuleLoader"
]

class ModuleLoader(importlib.abc.SourceLoader):
    """An ExtensionLoader creates a Loader object that can load a python module from a file path

    """

    def __init__(self, rootdir='', rootpkg = None):
        """
        :param rootdir: root directory to look for files off of
        :type rootdir: str
        :param rootpkg: root package to look for files off of
        :type rootpkg: str or None
        """
        self._dir=rootdir
        self._pkg = rootpkg
        super().__init__()

    def get_data(self, file):
        with open(file,'rb') as src:
            return src.read()

    def get_filename(self, fullname):
        if not os.path.exists(fullname):
            basename, ext = os.path.splitext(fullname.split(".")[-1])
            if ext != "":
                fullname = os.path.join(self._dir, basename+ext)
            else:
                fullname = os.path.join(self._dir, basename+".py")
                if not os.path.exists(fullname):
                    fullname = os.path.join(self._dir, basename)
        if os.path.isdir(fullname):
            fullname = os.path.join(fullname, "__init__.py")
        return fullname

    def get_spec(self, file, pkg = None):
        base_name = os.path.splitext(os.path.basename(file))[0]
        package_origin = file
        if pkg is None:
            pkg = self._pkg
        if pkg is None:
            raise ImportError("{}: package name required to load file".format(type(self)))
        if pkg == "":
            package_name = base_name
        else:
            package_name = pkg + "." + base_name
        spec = importlib.util.spec_from_loader(
            package_name,
            self,
            origin=package_origin
        )
        return spec

    def load(self, file, pkg = None):
        """loads a file as a module with optional package name

        :param file:
        :type file: str
        :param pkg:
        :type pkg: str or None
        :return:
        :rtype: module
        """
        d = self._dir
        try:
            if os.path.exists(file):
                self._dir = os.path.dirname(file)
            # print(file, self._dir)
            spec = self.get_spec(file, pkg)
            module = importlib.util.module_from_spec(spec)
            if module is None:
                module = importlib.util.module_from_spec(None)
            self.exec_module(module)
            sys.modules["DynamicImports."+module.__name__] = module
        finally:
            self._dir = d
        return module

