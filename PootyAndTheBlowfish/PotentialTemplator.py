"""
Defines classes to take a potential .so file an create a python extension that can wrap it and call into the potential
"""

from .Templator import TemplateWriter
import os, shutil

__all__ = [
    "PotentialTemplate",
    "PotentialTemplateError"
]

class PotentialTemplateError(ValueError):
    @classmethod
    def raise_bad_option(cls, parent, option, value):
        raise cls("{}: template option '{}' can't be {}".format(
            parent.__name__,
            option,
            value
        ))

    @classmethod
    def raise_missing_option(cls, parent, option, spec = None):
        """

        :param parent: parent class
        :type parent: type
        :param option:
        :type option: str
        :param spec:
        :type spec: str | None
        :return:
        :rtype:
        """
        if spec is None:
            raise cls("{}: template option '{}' needs a value".format(
                parent.__name__,
                option
            ))
        else:
            raise cls("{}: template option '{}' should be {}".format(
                parent.__name__,
                option,
                spec
            ))


class PotentialTemplate(TemplateWriter):
    """
    A `TemplateWriter` that handles most of the necessary boiler plate to get a C++ potential to play nice with DoMyCode
    """

    def __init__(self,
                 *ignored,
                 lib_name = None,
                 function_name = None,
                 potential_source = None,
                 linked_libs = None,
                 library_dirs = None,
                 macros = (),
                 requires_make = False,
                 custom_build = False,
                 raw_array_potential = False,
                 compile_on_build = True,
                 version = "1.0"
                 ):

        if len(ignored) > 0:
            raise PotentialTemplateError("{} requires all arguments to be passed as keywords".format(
                type(self).__name__
            ))

        if function_name is None:
            PotentialTemplateError.raise_missing_option(
                type(self),
                'function_name',
                "the name of the function used to get the potential value"
            )

        if linked_libs is None:
            linked_libs = [ lib_name ]
        if library_dirs is None:
            library_dirs = [ "." ]

        self.name = lib_name
        self.potential_source = potential_source
        self.function_name = function_name
        self.old_style_potential = raw_array_potential
        self.requires_make = requires_make
        self.custom_build = custom_build
        self.libs = linked_libs
        self.lib_dirs = library_dirs
        self.compile_on_build = compile_on_build

        super().__init__(
            os.path.join(os.path.dirname(__file__), "Templates", "PotentialTemplate"),
            LibName = lib_name,
            LibNameOfPotential = function_name,
            LinkedLibs = ", ".join("'{}'".format(x) for x in self.lib_names),
            LibDirs = ", ".join("'{}'".format(x) for x in library_dirs),
            LibMacros = ", ".join("'{}'".format(x) for x in macros),
            LibRequiresMake = requires_make,
            LibCustomBuild = custom_build,
            LibVersion = version,
            OldStylePotential = raw_array_potential
        )

    @property
    def lib_names(self):
        lib_names = []
        for l in self.libs:
            lib_name = l  # type: str
            if lib_name.endswith(".so") or lib_name.endswith("dll"):
                lib_name = os.path.splitext(os.path.basename(l))[0]
            lib_name = lib_name.split("lib")[-1]
            lib_names.append(lib_name)

        return lib_names

    def apply(self, out_dir):
        self.iterate_write(out_dir)
        src = self.potential_source
        if isinstance(src, str):
            dest = os.path.join(out_dir, self.name, "libs", os.path.basename(src))
            if os.path.isdir(src):
                if not os.path.isdir(dest): # should I raise an error if it already exists....???/
                    shutil.copytree(src, dest)
            else:
                shutil.copy(src, dest)

        if self.compile_on_build:
            import sys
            try:
                sys.path.insert(0, out_dir)
                env = {}
                exec("from {} import Potential".format(self.name), env, env)
            except ImportError:
                pass
            else:
                env["Potential"].load_potential()
            finally:
                try:
                    sys.path.remove(out_dir)
                except ValueError:
                    pass


