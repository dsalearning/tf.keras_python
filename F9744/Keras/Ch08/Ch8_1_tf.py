from tensorflow.keras.datasets import mnist

# 載入 MNIST 資料集, 如果是第一次載入會自行下載資料集
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 顯示 Numpy 二維陣列內容
print(X_train[0])
print(Y_train[0])   # 標籤資料

import matplotlib.pyplot as plt

plt.imshow(X_train[0], cmap="gray")
plt.title("Label: " + str(Y_train[0]))
plt.axis("off")
# 顯示數字圖片
plt.show()
