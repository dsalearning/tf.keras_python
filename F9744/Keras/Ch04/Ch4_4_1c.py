import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x = np.arange(-6, 6, 0.1)

plt.plot(x, tanh(x))
plt.title("tanh function")
plt.show()
