from matplotlib import pyplot as plt
from utils import load_pickle, convolve

def plot(filename, N= 10):
    data = load_pickle(filename)
    plt.plot(convolve(data['mean'], N), label='mean')
    plt.plot(convolve(data['max'], N), label='max')
    #plt.plot(convolve(data['diversity'], N), label='diversity')
    plt.legend()



plot('logs.pt')
plt.show()

print(sum(1 / (10*x) for x in range(1, 11)))