from matplotlib import pyplot as plt
from utils.utils import load_pickle, convolve

def plot(filename, N= 10):
    data = load_pickle(filename)
    plt.subplot(1, 2, 1)
    plt.plot(convolve(data['mean'], N), label='mean')
    plt.plot(convolve(data['max'], N), label='max')
    plt.subplot(1, 2, 2)
    plt.plot(convolve(data['diversity'], N), label='diversity')
    plt.legend()



plot('../logs.pt')
plt.show()
