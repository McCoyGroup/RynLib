import numpy as np, enum

__all__ = [
    "WalkerMask"
]

class WalkerMask:
    """
    A class for tracking mask info to apply to walkers
    """

    class Masks(enum.Enum):
        Unmarked = 0
        WeightTooLow = 1
        WeightTooHigh = 2
        EnergyTooLow = 3

    Unmarked = Masks.Unmarked
    WeightTooLow = Masks.WeightTooLow
    WeightTooHigh = Masks.WeightTooHigh
    EnergyTooLow = Masks.EnergyTooLow

    def __init__(self, walkers):
        self.mask = np.full(walkers.weights.shape, self.Masks.Unmarked.value, dtype=int)
        super().__init__()

    def reset(self):
        self.mask[:] = self.Masks.Unmarked.value

    def __setitem__(self, key, value):
        if isinstance(value, self.Masks):
            value = self.Masks.value
        self.mask[key] = value

    def where(self, value):
        if isinstance(value, self.Masks):
            value = self.Masks.value
        return np.where(self.mask == value)
