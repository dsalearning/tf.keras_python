from keras.models import Model
from keras.layers import Dense, Input
 
# 定義模型
model_input = Input(shape = (784,))
dense1 = Dense(512, activation="relu")(model_input)
dense2 = Dense(128, activation="relu")(dense1)
dense3 = Dense(32, activation ="relu")(dense2)
# 第 1 個分類輸出
output = Dense(10, activation="softmax")(dense3)
# 第 2 個自編碼器輸出
up_dense1 = Dense(128, activation="relu")(dense3)
up_dense2 = Dense(512, activation="relu")(up_dense1)
decoded_outputs = Dense(784)(up_dense2)
# 定義多輸出模型
model = Model(model_input, [output, decoded_outputs])
model.summary()   # 顯示模型摘要資訊
from keras.utils import plot_model

plot_model(model, to_file="Ch16_5_2.png", show_shapes=True)
