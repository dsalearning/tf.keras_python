import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
# 指定亂數種子
seed = 7
np.random.seed(seed)
# 載入資料集
(X_train, Y_train), (_, _) = mnist.load_data()
# 將圖片轉換成 4D 張量
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_train = X_train / 255
# 建立Keras的Sequential模型
model = Sequential()
print("載入模型...")
model = load_model("mnist.h5")
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 顯示各神經層
print("神經層數: ", len(model.layers))
for i in range(len(model.layers)):
    print(i, model.layers[i].name)
# 第2個 Conv2D 的 filters 形狀
print(model.layers[2].get_weights()[0].shape)
# 繪出第2個 Conv2D 的 32 個 filters
weights = model.layers[2].get_weights()[0]
for i in range(32):
    plt.subplot(6,6,i+1)
    plt.imshow(weights[:,:,0,i], cmap="gray", 
               interpolation="none")
    plt.axis("off")    
    

     