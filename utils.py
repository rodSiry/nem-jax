import pickle
import numpy as np


def save_pickle(obj, filename):
    F = open(filename, "wb")
    pickle.dump(obj, F)
    F.close()


def load_pickle(filename):
    F = open(filename, "rb")
    obj = pickle.load(F)
    F.close()
    return obj

def convolve(x, N=10):
    return np.convolve(x, np.ones(N) / N, mode="valid")
