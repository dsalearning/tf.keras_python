import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# 指定亂數種子
seed = 7
np.random.seed(seed)

# 載入訓練資料與測試資料
df_train = pd.read_csv("./titanic_train.csv")
df_test = pd.read_csv("./titanic_test.csv")

print(df_train.shape)
print(df_train.head())
print(df_test.shape)
print(df_test.head())

# 轉成Numpy陣列
dst_train = df_train.values
dst_test = df_test.values

# 分割成特徵資料與標籤資料
X_train = dst_train[:,:9]
Y_train = dst_train[:,9]
X_test = dst_test[:,:9]
Y_test = dst_test[:,9]

# 正規化
X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)

# 定義模型
model = Sequential()
model.add(Dense(11,input_dim=X_train.shape[1],activation="relu"))
model.add(Dense(11,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

# 編譯模型
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 訓練模型
history = model.fit(X_train,
                    Y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=10,
                    verbose=1)

# 評估模型
loss, accuracy = model.evaluate(X_train, Y_train)
print("訓練資料集的準確度： {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("測試資料集的準確度： {:.2f}".format(accuracy))

# 顯示Loss趨勢圖
import matplotlib.pyplot as plt

loss = history.history["loss"]
epochs = range(1,len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs,loss,"b-",label="loss")
plt.plot(epochs,val_loss,"r",label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 顯示Accuracy趨勢圖
import matplotlib.pyplot as plt

acc = history.history["acc"]
epochs = range(1,len(acc)+1)
val_acc = history.history["val_acc"]
plt.plot(epochs,acc,"b-",label="acc")
plt.plot(epochs,val_acc,"r",label="val_acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 訓練週期約18次後就過度擬合了, 再次訓練模型
history = model.fit(X_train,
                    Y_train,
                    validation_split=0.2,
                    epochs=18,
                    batch_size=10,
                    verbose=1)

# 再次查看圖型

# 用全部的資料重新訓練, 不用驗證資料
model.fit(X_train,
          Y_train,
          epochs=18,
          batch_size=10,
          verbose=0)

# 評估模型
loss, accuracy = model.evaluate(X_train, Y_train)
print("訓練資料集的準確度： {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("測試資料集的準確度： {:.2f}".format(accuracy))

# 儲存模型結構與權重
print("Saving Model: titanic.h5...")
model.save("titanic.h5")