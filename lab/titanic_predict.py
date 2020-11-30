import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.utils import to_categorical


seed = 7
np.random.seed(seed)

# 載入訓練資料與測試資料
df_train = pd.read_csv("./titanic_train.csv")
df_test = pd.read_csv("./titanic_test.csv")

# 轉成Numpy 陣列
dst_train = df_train.values
dst_test = df_test.values

# 區分特徵與標籤資料
X_train = dst_train[:,:9]
Y_train = dst_train[:,9]
X_test = dst_test[:,:9]
Y_test = dst_test[:,9]

# 正規化
X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)

# 定義模型與載入
model = Sequential()
model = load_model("titanic.h5")

# 編譯模型
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 評估模型
loss, accuracy = model.evaluate(X_train,Y_train)
print("\n訓練資料集的準確度： {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test,Y_test)
print("\n測試資料集的準確度： {:.2f}".format(accuracy))

# 怎麼知道訓練出來後的預測值 ？
Y_train_pred = model.predict_classes(X_train)

# 預測是否生存
Y_pred = model.predict_classes(X_test)
print(Y_pred[:,0])
print(Y_test.astype(int))

# 使用混淆矩陣進行分析
tb = pd.crosstab(Y_test.astype(int),
                 Y_pred[:,0],
                 rownames=["lable"],
                 colnames=["predict"])

print(tb)


print(df_test)
df_test["predict"] = Y_pred[:,0]



