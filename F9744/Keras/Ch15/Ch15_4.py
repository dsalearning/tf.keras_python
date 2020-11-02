import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical

# 指定亂數種子
seed = 7
np.random.seed(seed)
# 載入資料集
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 建立2個資料集，一個數字小於 5，一個數字大於等於 5
X_train_lt5 = X_train[Y_train < 5]
Y_train_lt5 = Y_train[Y_train < 5]
X_test_lt5 = X_test[Y_test < 5]
Y_test_lt5 = Y_test[Y_test < 5]

X_train_gte5 = X_train[Y_train >= 5]
Y_train_gte5 = Y_train[Y_train >= 5] - 5
X_test_gte5 = X_test[Y_test >= 5]
Y_test_gte5 = Y_test[Y_test >= 5] - 5
# 將圖片轉換成 4D 張量
X_train_lt5 = X_train_lt5.reshape(
        (X_train_lt5.shape[0], 28, 28, 1)).astype("float32")
X_test_lt5 = X_test_lt5.reshape(
        (X_test_lt5.shape[0], 28, 28, 1)).astype("float32")
X_train_gte5 = X_train_gte5.reshape(
        (X_train_gte5.shape[0], 28, 28, 1)).astype("float32")
X_test_gte5 = X_test_gte5.reshape(
        (X_test_gte5.shape[0], 28, 28, 1)).astype("float32")
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_train_lt5 = X_train_lt5 / 255
X_test_lt5 = X_test_lt5 / 255
X_train_gte5 = X_train_gte5 / 255
X_test_gte5 = X_test_gte5 / 255
# One-hot編碼
Y_train_lt5 = to_categorical(Y_train_lt5, 5)
Y_test_lt5 = to_categorical(Y_test_lt5, 5)
Y_train_gte5 = to_categorical(Y_train_gte5, 5)
Y_test_gte5 = to_categorical(Y_test_gte5, 5)
# 定義模型
model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3),
                 input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(5, activation="softmax"))
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
history = model.fit(X_train_lt5, Y_train_lt5, validation_split=0.2,
                    epochs=5, batch_size=128, verbose=2)
# 評估模型
loss, accuracy = model.evaluate(X_test_lt5, Y_test_lt5)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 顯示各神經層
print(len(model.layers))
for i in range(len(model.layers)):
    print(i, model.layers[i])
# 凍結上層模型
for i in range(4):
    model.layers[i].trainable = False
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])    
# 訓練模型
history = model.fit(X_train_gte5, Y_train_gte5, validation_split=0.2,
                    epochs=5, batch_size=128, verbose=2)
# 評估模型
loss, accuracy = model.evaluate(X_test_gte5, Y_test_gte5)
print("測試資料集的準確度 = {:.2f}".format(accuracy))