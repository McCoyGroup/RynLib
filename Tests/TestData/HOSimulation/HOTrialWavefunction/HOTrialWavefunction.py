import numpy as np

omega=1
m=1
hbar=1

def trial_wavefunction(coords, atoms, extra_args):
    floats = extra_args[2]
    if len(floats) > 0:
        re = floats[0]
    else:
        re = 0.9
    main_shape = coords.shape[:-2]
    coords = coords.reshape((np.prod(main_shape),) + coords.shape[-2:])
    rs = np.linalg.norm(coords[:, 0] - coords[:, 1], axis=1)
    psi_vals = np.exp(-(rs-re)**2)
    return psi_vals.reshape(main_shape)