import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)  # 指定亂數種子
# 載入波士頓房屋資料集
df = pd.read_csv("./boston_housing.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:13]
Y = dataset[:, 13]
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割訓練和測試資料集
X_train, Y_train = X[:404], Y[:404]     # 訓練資料前404筆
X_test, Y_test = X[404:], Y[404:]       # 測試資料後102筆
# 定義模型
def build_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), activation="relu"))
    model.add(Dense(1))
    # 編譯模型
    model.compile(loss="mse", optimizer="adam", 
                  metrics=["mae"])
    return model

k = 4
nb_val_samples = len(X_train) // k
nb_epochs = 80
mse_scores = []
mae_scores = []
for i in range(k):
    print("Processing Fold #" + str(i))
    # 取出驗證資料集
    X_val = X_train[i*nb_val_samples: (i+1)*nb_val_samples]
    Y_val = Y_train[i*nb_val_samples: (i+1)*nb_val_samples]
    # 結合出訓練資料集
    X_train_p = np.concatenate(
            [X_train[:i*nb_val_samples],
            X_train[(i+1)*nb_val_samples:]], axis=0)
    Y_train_p = np.concatenate(
            [Y_train[:i*nb_val_samples],
            Y_train[(i+1)*nb_val_samples:]], axis=0)
    model = build_model()
    # 訓練模型
    model.fit(X_train_p, Y_train_p, epochs=nb_epochs, 
              batch_size=16, verbose=0)
    # 評估模型
    mse, mae = model.evaluate(X_val, Y_val)
    mse_scores.append(mse)
    mae_scores.append(mae)
    
print("MSE_val: ", np.mean(mse_scores))
print("MAE_val: ", np.mean(mae_scores))
# 使用測試資料評估模型
mse, mae = model.evaluate(X_test, Y_test)    
print("MSE_test: ", mse)
print("MAE_test: ", mae)
