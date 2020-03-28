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

class PotentialFunction:
    def __init__(self,
                 lib_name = None,
                 name = None,
                 pointer = None,
                 arguments = (),
                 default_args = None,
                 old_style_potential = None,
                 return_type = "Real_t"
                 ):
        self.lib_name = lib_name
        self.name = name
        self.pointer = pointer # name in the PyCapsule
        self.arguments = [PotentialArgument(*x) for x in arguments]
        if default_args is None:
            arg_names = [a[0] for a in arguments]
            if "coords" in arg_names or "atoms" in arg_names or "raw_coords" in arg_names or "raw_atoms" in arg_names:
                default_args = False
            else:
                default_args = True
        self.default_args = default_args
        if old_style_potential is None:
            arg_names = [a[0] for a in arguments]
            if "raw_coords" in arg_names or "raw_atoms" in arg_names:
                old_style_potential = True
            else:
                old_style_potential = False
        self.old_style = old_style_potential
        self.return_type = return_type

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
        if self.default_args:
            main_args = ["coords", "atoms"] if not self.old_style else ["raw_coords", "raw_atoms"]
        else:
            main_args = []
        return self.name + "(" + ",".join(
            main_args + [str(arg) for arg in self.arguments]
        ) + ")"

    def get_potential_declaration(self):
        if self.default_args:
            main_args = ["Coordinates", "Names"] if not self.old_style else ["RawWalkerBuffer", "const char*"]
        else:
            main_args = []
        def get_arg_type(arg):
            if arg.name == "coords":
                return "Coordinates"
            elif arg.name == "raw_coords":
                return "RawWalkerBuffer"
            elif arg.name == "atoms":
                return "Names"
            elif arg.name == "raw_atoms":
                return "const char*"
            elif isinstance(arg.dtype, str):
                return arg.dtype
            else:
                return arg.dtype.__name__
        return self.name + "(" + ",".join(
            main_args + [ get_arg_type(arg) for arg in self.arguments ]
        ) + ")"

    def get_attach_declaration(self):
        return """if (!_AttachCapsuleToModule(m, {lib}_{name}Wrapper, "{pointer}")) {{ return NULL; }}""".format(
            lib = self.lib_name,
            name = self.name,
            pointer = self.pointer
        )

    old_style_block = """
            // Copy data into a single array
            int nels = coords.size() * coords[0].size();
            std::vector<Real_t> long_vec (nels);
            nels = 0;
            std::vector<Real_t> sub_vec;
            for (unsigned long v=0; v<coords.size(); v++) {
                sub_vec = coords[v];
                long_vec.insert(long_vec.end(), sub_vec.begin(), sub_vec.end());
            }
            RawWalkerBuffer raw_coords = long_vec.data();
        
            std::string long_atoms;
            for (unsigned long v=0; v<atoms.size(); v++) {
                long_atoms = long_atoms + atoms[v] + " ";
            }
            const char* raw_atoms = long_atoms.data();
        """

    def get_wrapper(self):
        return """
        {return_type} {lib}_{name}(
            Coordinates coords,
            Names atoms,
            ExtraBools extra_bools,
            ExtraInts extra_ints,
            ExtraFloats extra_floats
            ) {{
        
            // Load extra args (if necessary)
            {load_bools}
            {load_ints}
            {load_floats}
            {old_style_block}
        
            return {potential_call};
        }}
        
        static PyObject* {lib}_{name}Wrapper = PyCapsule_New((void *){lib}_{name}, "{pointer}", NULL);
        """.format(
            lib = self.lib_name,
            name = self.name,
            return_type = self.return_type,
            load_bools = self.get_extra_args_call(bool),
            load_ints=self.get_extra_args_call(int),
            load_floats=self.get_extra_args_call(float),
            old_style_block = self.old_style_block if self.old_style else "",
            pointer = self.pointer,
            potential_call = self.get_potential_call()
        )

    def get_declaration(self):
        return """
        {return_type} {lib}_{name}(
            const Coordinates,
            const Names,
            const ExtraBools,
            const ExtraInts,
            const ExtraFloats
            );
        {return_type} {declaration};""".format(
            return_type=self.return_type,
            lib=self.lib_name,
            name=self.name,
            declaration=self.get_potential_declaration()
        )

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
                 raw_array_potential = None,
                 arguments = (),
                 linked_libs = None,
                 static_source = False,
                 extra_functions = ()
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
        :param extra_functions: Extra functions if we'll have more than the default one
        :type extra_functions:
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
        main_function = PotentialFunction(
            lib_name = lib_name,
            name = function_name,
            old_style_potential=raw_array_potential,
            arguments=arguments,
            pointer="_potential"
        )

        self.functions = [main_function] + [
            o if isinstance(o, PotentialFunction) else PotentialFunction(**o) for o in extra_functions
        ]

        self.libs = linked_libs
        self.static_source = static_source

        super().__init__(
            os.path.join(os.path.dirname(__file__), "Templates", "PotentialTemplate"),
            LibName = lib_name,
            LIBNAME = lib_name.upper(),
            MethodWrappers = "\n\n".join(g.get_wrapper() for g in self.functions),
            AttachMethods="\n".join(g.get_attach_declaration() for g in self.functions),
            MethodDeclarations="\n".join(g.get_declaration() for g in self.functions)
        )

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


