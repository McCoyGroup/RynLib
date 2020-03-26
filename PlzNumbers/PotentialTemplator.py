"""
Defines classes to take a potential .so file an create a python extension that can wrap it and call into the potential
"""

from ..RynUtils.TemplateWriter import TemplateWriter
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

class PotentialArgument:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype
    def __iter__(self):
        for a in (self.name, self.dtype):
            yield a
    def __str__(self):
        return self.name

class PotentialTemplate(TemplateWriter):
    """
    A `TemplateWriter` that handles most of the necessary boiler plate to get a C++ potential to play nice with DoMyCode
    """
    __props__ = [
        "lib_name",
        "function_name",
        "potential_source",
        "raw_array_potential",
        "arguments",
        "linked_libs"
    ]
    def __init__(self,
                 *ignored,
                 lib_name = None,
                 function_name = None,
                 potential_source = None,
                 raw_array_potential = False,
                 arguments = (),
                 linked_libs = None,
                 static_source = False
                 ):
        """

        :param lib_name:
        :type lib_name: str
        :param function_name:
        :type function_name:
        :param potential_source:
        :type potential_source:
        :param raw_array_potential:
        :type raw_array_potential:
        :param arguments:
        :type arguments:
        :param linked_libs:
        :type linked_libs:
        """

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

        self.name = lib_name
        self.potential_source = potential_source
        self.function_name = function_name
        self.old_style_potential = raw_array_potential
        self.arguments = [PotentialArgument(*x) for x in arguments]
        self.libs = linked_libs

        self.static_source = static_source

        super().__init__(
            os.path.join(os.path.dirname(__file__), "Templates", "PotentialTemplate"),
            LibName = lib_name,
            LIBNAME = lib_name.upper(),
            PotentialCall = self.get_potential_call(),
            PotentialCallDeclaration = self.get_potential_declaration(),
            OldStylePotential = "true" if self.old_style_potential else "false",
            PotentialLoadExtraBools=self.get_extra_args_call(bool),
            PotentialLoadExtraInts=self.get_extra_args_call(int),
            PotentialLoadExtraFloats=self.get_extra_args_call(float)
        )

    def get_extra_args_call(self, dtype):
        """

        :param dtype:
        :type dtype: type
        :return:
        :rtype:
        """
        n=0
        call=[]
        for arg in self.arguments:
            if arg.dtype is dtype or (isinstance(arg.dtype, str) and arg.dtype == dtype.__name__):
                call.append("{dtype} {name} = extra_{dtype}s[{n}];".format(
                    name = arg.name,
                    dtype = dtype.__name__,
                    n = n
                ))
                n+=1
        return "\n".join(call)

    def get_potential_call(self):
        main_args = ["coords", "atoms"] if not self.old_style_potential else ["raw_coords", "raw_atoms"]
        return self.function_name + "(" + ",".join(
            main_args + [str(arg) for arg in self.arguments]
        ) + ")"

    def get_potential_declaration(self):
        main_args = ["Coordinates", "Names"] if not self.old_style_potential else ["RawWalkerBuffer", "const char*"]
        return self.function_name + "(" + ",".join(
            main_args + [arg.dtype if isinstance(arg.dtype, str) else arg.dtype.__name__ for arg in self.arguments]
        ) + ")"

    # @property
    # def lib_names(self):
    #     lib_names = []
    #     for l in self.libs:
    #         lib_name = l  # type: str
    #         if lib_name.endswith(".so") or lib_name.endswith("dll"):
    #             lib_name = os.path.splitext(os.path.basename(l))[0]
    #         lib_name = lib_name.split("lib")[-1]
    #         lib_names.append(lib_name)
    #
    #     return lib_names

    def apply(self, out_dir):
        self.iterate_write(out_dir)
        if not self.static_source:
            src = self.potential_source
            if isinstance(src, str):
                dest_dir = os.path.join(out_dir, self.name, "libs")
                dest = os.path.join(dest_dir, os.path.basename(src))
                if os.path.isdir(src):
                    if not os.path.isdir(dest): # should I raise an error if it already exists....???/
                        shutil.copytree(src, dest)
                else:
                    os.makedirs(dest_dir)
                    shutil.copy(src, dest)


