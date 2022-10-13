import numpy as np


class IOOptions:

    @classmethod
    def default(cls):
        return IOOptions(True, True, "<f4", intensity="normalized", reduce_to=None, filetype=".ply", custom_index=False)

    @classmethod
    def only_localization(cls):
        return IOOptions(binary=False, local=True, xyz_accuracy=None, intensity=None, reduce_to=None, filetype=None, custom_index=False)

    @classmethod
    def do_nothing(cls):
        return IOOptions(binary=False, local=False, xyz_accuracy=None, intensity=None, reduce_to=None, filetype=None, custom_index=False)

    def __init__(self, binary, local, xyz_accuracy, intensity, reduce_to, filetype, custom_index):
        self.compression = "uncompressed"
        self.binary = binary
        self.local_system = local
        self.xyz_accuracy = xyz_accuracy
        self.reduce_to = reduce_to
        self.intensity = intensity
        self.filetype = filetype
        self.custom_index = custom_index
        self.shift_system = []
