import os
import struct
import numpy as np


def load_mnist(kind='train'):
    """Load MNIST data from `path`"""
    with open('%s-labels.idx1-ubyte' % kind, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open('%s-images.idx3-ubyte' % kind, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


load_mnist()
