import numpy as np


__all__ = [
    "Constants"
]

class Constants:
    '''
    A dumb little class that handles basic unit conversions and stuff.
    It's gotten popular in the group, though, since it's so simple
    '''
    atomic_units = {
        "wavenumbers" : 4.55634e-6,
        "angstroms" : 1/0.529177,
        "amu" : 1.000000000000000000/6.02213670000e23/9.10938970000e-28   #1822.88839  g/mol -> a.u.
    }

    masses = {
        "H" : ( 1.00782503223, "amu"),
        "O" : (15.99491561957, "amu")
    }

    @classmethod
    def convert(cls, val, unit, in_AU = True):
        vv = cls.atomic_units[unit]
        return (val * vv) if in_AU else (val / vv)

    @classmethod
    def mass(cls, atom, in_AU = True):
        m = cls.masses[atom]
        if in_AU:
            m = cls.convert(*m)
        return m

    # this really shouldn't be here........... but oh well
    water_structure = (
        ["O", "H", "H"],
        np.array(
            [
                [0.0000000,0.0000000,0.0000000],
                [0.9578400,0.0000000,0.0000000],
                [-0.2399535,0.9272970,0.0000000]
            ]
        )
    )