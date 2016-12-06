# Data Parsing
import struct as struct
from itertools import izip_longest

def read_labels(path):
    labels_file = open(path, "rb")
    magic_nr, size = struct.unpack(">II", labels_file.read(8))
    labels = struct.unpack(">" + "B" * size, labels_file.read())
    return labels

def each_slice(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def read_images(path):
    images_file = open(path, "rb")
    magic_nr, size, rows, cols = struct.unpack(">IIII", images_file.read(16))
    images = struct.unpack(">" + "B" * size * rows * cols, images_file.read())
    images = list(each_slice(images, 28 * 28))
    return images

def read_train_labels():
    return read_labels('mnist/train-labels-idx1-ubyte')

def read_train_images():
    return read_images('mnist/train-images-idx3-ubyte')

def read_test_labels():
    return read_labels('mnist/t10k-labels-idx1-ubyte')

def read_test_images():
    return read_images('mnist/t10k-images-idx3-ubyte')


