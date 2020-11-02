import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense

np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
df = pd.read_csv("./diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
Y = dataset[:, 8]
# 定義模型 - 使用 Function API
inputs = Input(shape=(8,))
hidden1 = Dense(10, activation="relu")(inputs)
hidden2 = Dense(8, activation="relu")(hidden1)
outputs = Dense(1, activation="sigmoid")(hidden2)
model = Model(inputs=inputs, outputs=outputs)
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
# 訓練模型
model.fit(X, Y, epochs=15, batch_size=10)
# 評估模型
loss, accuracy = model.evaluate(X, Y)
print("準確度 = {:.2f}".format(accuracy))
