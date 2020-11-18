
import numpy as np

omega=1
m=1
hbar=1

def _potential(coords, atoms, extra_args):
    coords = np.array(coords)
    floats = extra_args[2]
    if len(floats) > 0:
        re = floats[0]
        k = floats[1]
    else:
        re = 0.9
        k = 1.0
    main_shape = coords.shape[:-2]
    ncoords=int(np.prod(main_shape))
    if coords.ndim > 2:
        coords = coords.reshape((ncoords,) + coords.shape[-2:])
    else:
        coords = coords[np.newaxis]
    rs = np.linalg.norm(coords[:, 0] - coords[:, 1], axis=1)
    pot = k/2*(rs-re)**2
    return pot.reshape(main_shape)