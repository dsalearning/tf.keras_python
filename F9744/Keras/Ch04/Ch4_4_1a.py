import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-6, 6, 0.1)

plt.plot(x, relu(x))
plt.title("relu function")
plt.show()
