import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

np.random.seed(7)  # 指定亂數種子
target_mapping = {"setosa": 0,
                  "versicolor": 1,
                  "virginica": 2}
# 載入資料集
df = pd.read_csv("./iris_data.csv")
df["target"] = df["target"].map(target_mapping)
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:,0:4].astype(float)
Y = to_categorical(dataset[:,4])
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割成訓練和測試資料集
X_train, Y_train = X[:120], Y[:120]     # 訓練資料前120筆
X_test, Y_test = X[120:], Y[120:]       # 測試資料後30筆
# 建立Keras的Sequential模型
model = Sequential()
model = load_model("iris.h5")
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 計算分類的預測值
print("\nPredicting ...")
Y_pred = model.predict_classes(X_test)
print(Y_pred)
Y_target = dataset[:,4][120:].astype(int)
print(Y_target)
# 顯示混淆矩陣
tb = pd.crosstab(Y_target, Y_pred, rownames=["label"], colnames=["predict"])
print(tb)
tb.to_html("Ch6_1_3.html")
