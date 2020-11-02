import numpy as np

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

x = np.array([1,2,3,4,1,2,3])

y = softmax(x)
print(y)
