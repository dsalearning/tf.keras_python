from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate

# 定義模型
# 第 1 個灰階圖片輸入
input1 = Input(shape=(28, 28, 1))
conv11 = Conv2D(16, (3,3), activation="relu")(input1)
pool11 = MaxPooling2D(pool_size=(2,2))(conv11)
conv12 = Conv2D(32, (3,3), activation="relu")(pool11)
pool12 = MaxPooling2D(pool_size=(2,2))(conv12)
flat1 = Flatten()(pool12)
# 第 2 個彩色圖片輸入
input2 = Input(shape=(28, 28, 3))
conv21 = Conv2D(16, (3,3), activation="relu")(input2)
pool21 = MaxPooling2D(pool_size=(2,2))(conv21)
conv22 = Conv2D(32, (3,3), activation="relu")(pool21)
pool22 = MaxPooling2D(pool_size=(2,2))(conv22)
flat2 = Flatten()(pool22)
# 合併 2 個輸入
merge = concatenate([flat1, flat2])
dense1 = Dense(512, activation="relu")(merge)
dense2 = Dense(128, activation="relu")(dense1)
dense3 = Dense(32, activation="relu")(dense2)
output = Dense(10, activation="softmax")(dense3)
# 定義多輸入模型
model = Model(inputs=[input1, input2], outputs=output)
model.summary()   # 顯示模型摘要資訊
from keras.utils import plot_model

plot_model(model, to_file="Ch16_5_1.png", show_shapes=True)
