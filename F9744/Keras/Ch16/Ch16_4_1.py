from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate

# 定義模型
shared_input = Input(shape=(64, 64, 1))
# 第1個共享輸入層的卷積和池化層
conv1 = Conv2D(32, kernel_size=3, activation="relu")(shared_input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flat1 = Flatten()(pool1)
# 第2個共享輸入層的卷積和池化層
conv2 = Conv2D(16, kernel_size=5, activation="relu")(shared_input)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)
# 合併 2 個共享輸入層的卷積和池化層
merge = concatenate([flat1, flat2])
hidden1 = Dense(10, activation="relu")(merge)
output = Dense(1, activation="sigmoid")(hidden1)
model = Model(inputs=shared_input, outputs=output)
model.summary()   # 顯示模型摘要資訊
from keras.utils import plot_model

plot_model(model, to_file="Ch16_4_1.png", show_shapes=True)
