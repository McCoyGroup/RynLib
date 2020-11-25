
"""
We want the individual potential specifications to look like

```
{
    "name":"ModuleName",
    "functions": [
            {
                "name":"function_name_1",
                "arguments":[
                    {
                        "name":"nthreads",
                        "type":"int"
                        },
                    {
                        "name":"coords",
                        "type":"float",
                        "shape":[0, 0, 3] # nwalkers x natoms x 3, but we don't know how many walkers or atoms
                        },
                    {
                        "name":"atoms",
                        "type":"str",
                        "shape":[0] # natoms, but we don't know how many walkers or atoms
                        }
                    ],
                "type": "double"
                # return type is implicitly `float` at this stage...
                },
            {
                "name":"function_name_2",
                "arguments":[
                    {
                        "name":"coords",
                        "type":"float",
                        "shape":[0, 3] # natoms x 3, but we don't know how many atoms
                        },
                    {
                        "name":"atoms",
                        "type":"str",
                        "shape":[0] # natoms, but we don't know how many walkers or atoms
                        }
                    ]
                # return type is implicitly `float` at this stage...
                }
    ]
}
```
"""

import collections

__all__ = [
    "PotentialModuleSpec",
    "PotentialFunctionSpec",
    "PotentialArgumentSpec"
]


class PotentialAtomsPattern:
    """
    Specifies an allowed set of atoms should be arranged
    """
    def __init__(self, atoms):
        self.atoms = atoms
    def validate(self, arg):
        ...


