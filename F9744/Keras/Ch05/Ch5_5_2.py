import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.utils import to_categorical

np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
df = pd.read_csv("./diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
Y = dataset[:, 8]
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# One-hot編碼
Y = to_categorical(Y)
# 分割訓練和測試資料集
X_train, Y_train = X[:690], Y[:690]     # 訓練資料前690筆
X_test, Y_test = X[690:], Y[690:]       # 測試資料後78筆
# 載入模型結構
from keras.models import model_from_json

model = Sequential()
with open("Ch5_5_1Model.config", "r") as text_file:
    json_str = text_file.read()
model = model_from_json(json_str)
# 載入權重
model.load_weights("Ch5_5_1Model.weight", by_name=False)
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=["accuracy"])
# 評估模型
loss, accuracy = model.evaluate(X_test, Y_test)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 測試資料集的預測值
predict_values = model.predict(X_test, batch_size=10, verbose=0)
print(predict_values[0])


