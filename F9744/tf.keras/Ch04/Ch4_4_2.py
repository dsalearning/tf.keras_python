import numpy as np

def MSE(y, t):
    return 0.5*np.sum((y-t)**2)

y_data = np.array([0.5,2.1,1.5,3.8,0.7,1.8,3.4])
t_data = np.array([1,2,3,4,1,2,3])

print(MSE(y_data, t_data))
