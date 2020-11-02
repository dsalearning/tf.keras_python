import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

seed = 7
np.random.seed(seed)
# 載入Titanic的測試資料集
df_test = pd.read_csv("./titanic_test.csv")
dataset_test = df_test.values
# 分割成特徵資料和標籤資料
X_test = dataset_test[:, 0:9]
Y_test = dataset_test[:, 9]
# 特徵標準化
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)
# 建立Keras的Sequential模型
model = Sequential()
model = load_model("titanic.h5")
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 計算分類的預測值
print("\nPredicting ...")
Y_pred = model.predict_classes(X_test)
print(Y_pred[:,0])
print(Y_test.astype(int))
# 顯示混淆矩陣
tb = pd.crosstab(Y_test.astype(int), Y_pred[:,0],
                 rownames=["label"], colnames=["predict"])
print(tb)
tb.to_html("Ch6_2_4.html")