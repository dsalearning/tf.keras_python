import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# 指定亂數種子
seed = 10
np.random.seed(seed)
# 載入資料集
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# 打亂 2 個 Numpy 陣列
def randomize(a, b):
    permutation = list(np.random.permutation(a.shape[0]))
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    
    return shuffled_a, shuffled_b

X_train, Y_train = randomize(X_train, Y_train)
# 取出部分的訓練資料
X_train = X_train[:500]
Y_train = Y_train[:500]
# 將訓練資料的圖片尺寸放大
print("將訓練資料的圖片尺寸放大...")
X_train_new = np.array(
  [np.asarray(Image.fromarray(X_train[i]).resize(
          (200, 200))) for i in range(0, len(X_train))])
# 繪出6張圖片
fig = plt.figure(figsize=(10,7))
sub_plot= 230
for i in range(0, 6):
    ax = plt.subplot(sub_plot+i+1)
    ax.imshow(X_train_new[i], cmap="binary")
    ax.set_title("Label: " + str(Y_train[i]))

plt.show()