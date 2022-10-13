import numpy as np
import pandas as pd


def apply_augmentations(data, augmentations):
    tmp = pd.options.mode.chained_assignment = None
    pd.options.mode.chained_assignment = None
    for iname, augmentation in augmentations.items():
        data[iname] = augmentation.augment(data[iname])
    pd.options.mode.chained_assignment = tmp


class TypeAugmentation:
    def __init__(self, _type, augmentation):
        self.type = _type
        self.augmentation = augmentation

    def augment(self, data):
        return self.augmentation(data, self.type).astype(self.type)


TA = TypeAugmentation


class TrueValue:
    def __call__(self, data: np.ndarray, target_type):
        return data


class Normalize(TrueValue):
    def __init__(self, scale=1):
        self._scale = scale
        self.original_min = 0
        self.original_max = 0

    def __call__(self, data: np.ndarray, target_type):

        self.original_min = np.min(data)
        self.original_max = np.max(data)
        self._scale = self._scale / max(np.abs(self.original_min), np.abs(self.original_max))

        if np.issubdtype(target_type, np.integer):
            dtype_min = np.iinfo(target_type).min
            dtype_max = np.iinfo(target_type).max
            # scale if its a uint
            if self.original_min < 0 and dtype_min == 0:
                raise TypeError("Unsigned types cannot store negative numbers !")
            self._scale /= dtype_max

        if self._scale == np.inf:
            raise TypeError("Type casting creates a zero scaling !")

        return data * self._scale


class UnpackBinary(TrueValue):
    def __init__(self, _mask):
        self._mask = _mask

    def __call__(self, data: np.ndarray, target_type):
        return ((data & self._mask) / (self._mask & -self._mask)).astype(np.int8)


class Scale(TrueValue):
    def __init__(self, _min=0.0, _max=100.0):
        self._target_min = _min
        self._target_max = _max
        self._offset = 0
        self._scale = 1

    def __call__(self, data: np.ndarray, target_type):
        dmin = np.min(data)
        dmax = np.max(data)
        self._scale = (dmax - dmin) / (self._target_max - self._target_min)
        self._offset = dmin - self._target_min
        return (data * self._scale) - self._offset


class GobalOffset(TrueValue):
    def __init__(self, xyz_offset):
        self._offset = xyz_offset

    @property
    def offset(self):
        return self._offset

    def __call__(self, data: np.ndarray, target_type):
        return data - self._offset


class LocalSystem(TrueValue):
    def __init__(self):
        self._offset = 0
        self._scale = 1

    @property
    def offset(self):
        return self._offset

    def __call__(self, data: np.ndarray, target_type):
        dmin = np.min(data)
        dmax = np.max(data)
        self._offset = dmin + (dmax - dmin) / 2
        return data - self._offset


class Category(Scale):
    def __init__(self):
        pass

    def __call__(self, data, target_type):
        return data
