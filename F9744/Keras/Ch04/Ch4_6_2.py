import numpy as np

def normalization(raw):
    max_value = max(raw)
    min_value = min(raw)
    norm = [(float(i)-min_value)/(max_value-min_value) for i in raw]
    return norm
    
x = np.array([255, 128, 45, 0])

print(x)
norm = normalization(x)
print(norm)
print(x/255)