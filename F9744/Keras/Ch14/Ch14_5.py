import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
df = pd.read_csv("./diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
Y = dataset[:, 8]
# 定義模型
model = Sequential()
model.add(Dense(10, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()  # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
# 訓練模型
from keras.callbacks import Callback
# 繼承 Callback 類別建立 fitHistory 類別
class fitHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.acc.append(logs.get("acc"))
        self.losses.append(logs.get('loss'))
# 建立 fitHistory 物件
history = fitHistory()
model.fit(X, Y, batch_size=64, epochs=5, verbose=0,
          callbacks=[history])

print("筆數: ", len(history.acc))
print(history.acc)
print("筆數: ", len(history.losses))
print(history.losses)
# 評估模型
loss, accuracy = model.evaluate(X, Y)
print("準確度 = {:.2f}".format(accuracy))
