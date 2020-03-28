import numpy as np

# I should pull in a bunch of what I collected from McUtils...

__all__ = [
    "Constants"
]

class Constants:
    '''
    A dumb little class that handles basic unit conversions and stuff.
    It's gotten popular in the group, though, since it's so simple
    '''
    atomic_units = {
        "wavenumbers" : 1/219474.6313631999788043049,
        "angstroms" : 1/0.52917721090299998131181701668799047776,
        "amu" : 1.000000000000/6.02213670000e23/9.10938970000e-28   #1822.88839  g/mol -> a.u.
    }

    masses = {
        "H": (1.007825032230, "amu"),
        "D": (2.014101778120, "amu"),
        "T": (3.016049277900, "amu"),
        "C": (12.00000000000, "amu"),
        "O": (15.99491561957, "amu"),
        "N": (14.00307400443, "amu"),
        "F": (18.99840316273, "amu"),
        "Cl":  (34.968852682, "amu")
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