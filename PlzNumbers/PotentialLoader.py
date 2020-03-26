from ..RynUtils import CLoader

__all__ = [
    "PotentialLoader"
]

class PotentialLoader:
    """
    Provides a standardized way to load and compile a potential using a potential template
    """
    __props__ = [
        "src_ext",
        "description",
        "version",
        "include_dirs",
        "linked_libs",
        "macros",
        "source_files",
        "build_script",
        "requires_make",
        "out_dir",
        "cleanup_build"
    ]
    def __init__(self,
                 name,
                 src,
                 src_ext='src',
                 description="A compiled potential",
                 version="1.0.0",
                 include_dirs=None,
                 linked_libs=None,
                 macros=None,
                 source_files=None,
                 build_script=None,
                 requires_make=False,
                 out_dir=None,
                 cleanup_build=True
                 ):
        self.loader = CLoader(name, src,
                              src_ext=src_ext,
                              description=description,
                              version=version,
                              include_dirs=include_dirs,
                              linked_libs=linked_libs,
                              macros=macros,
                              source_files=source_files,
                              build_script=build_script,
                              requires_make=requires_make,
                              out_dir=out_dir,
                              cleanup_build=cleanup_build
                              )
        self._lib = None

    @property
    def lib(self):
        if self._lib is None:
            self._lib = self.loader.load()
        return self._lib
    @property
    def pointer(self):
        return self.lib._potential