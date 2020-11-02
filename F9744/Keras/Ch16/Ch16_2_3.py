import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
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
model = load_model("mnist.h5")
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 使用 Sequential 建立 Conv2D 層
model_test = Sequential()
model_test.add(Conv2D(16, kernel_size=(5, 5), padding="same",
                 input_shape=(28, 28, 1), activation="relu"))
for i in range(len(model_test.layers)):
    model_test.layers[i].set_weights(model.layers[i].get_weights())
   
output = model_test.predict(X_train[0].reshape(1,28,28,1))
# 繪出第1個 Conv2D 層的輸出
plt.figure(figsize=(10,8))
for i in range(0,16):
    plt.subplot(4,4,i+1)
    plt.imshow(output[0,:,:,i], cmap="gray")
    plt.axis("off")

     