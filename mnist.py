import array
import numpy
import pickle


# 
# not the best programming practice ...
# 
f = open('mnist/stat.npy', 'rb')
ex, ex2 = pickle.load(f)
f.close()

data_mean = ex
data_var = ex2 - ex * ex


def load_images(filename):
    f = open(filename, 'rb')

    sig = f.read(4)
    dim1 = int.from_bytes(f.read(4), byteorder='big', signed=False)
    dim2 = int.from_bytes(f.read(4), byteorder='big', signed=False)
    dim3 = int.from_bytes(f.read(4), byteorder='big', signed=False)

    data = numpy.array(array.array('B', f.read()), dtype=float)
    result = data.reshape(dim1, dim2, dim3)

    f.close()

    return result


def load_labels(filename):
    f = open(filename, 'rb')

    sig = f.read(4)
    dim1 = int.from_bytes(f.read(4), byteorder='big', signed=False)

    result = numpy.array(array.array('B', f.read()))

    f.close()

    return result


def normalize(img):
    return (img - data_mean) / numpy.sqrt(data_var)

