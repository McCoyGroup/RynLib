from ..RynUtils import CLoader

__all__ = [
    "PotentialLoader"
]

class PotentialLoader:
    """
    Provides a standardized way to load and compile a potential using a potential template
    """
    def __init__(self,
                 name,
                 src,
                 **load_opts
                 ):
        self.loader = CLoader(name, src, **load_opts)
        self._lib = None

    @property
    def lib(self):
        if self._lib is None:
            self._lib = self.loader.load()
        return self._lib
    @property
    def pointer(self):
        return self.lib._potential