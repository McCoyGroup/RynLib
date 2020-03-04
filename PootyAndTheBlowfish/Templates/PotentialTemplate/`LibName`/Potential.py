"""
Defines the potential class for `LibName` in such a way as to be easily composable with DoMyCode
"""

class Potential:

    old_style_potential = `OldStylePotential`
    potential_holder = None

    def __init__(self):
        self.load_potential()

    @classmethod
    def load_potential(cls):
        if cls.potential_holder is None:
            cls.potential_holder = cls._compile_and_load()

    @classmethod
    def _compile_and_load(cls):
        import os
        lib_dir = os.path.dirname(__file__)
        lib_lib_dir = os.path.join(lib_dir, "libs")
        cur_dir = os.getcwd()

        try:
            os.chdir(lib_lib_dir)
            from .src.`LibName` import _potential
        except ImportError:
            from .src.setup import failed
            if failed:
                raise
            else:
                from .src.`LibName` import _potential
        finally:
            os.chdir(cur_dir)

        return _potential