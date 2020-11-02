from keras.models import Sequential
from keras.layers import Dense

# 定義模型
model = Sequential()
model.add(Dense(10, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()  # 顯示模型摘要資訊
# 繪出模型圖片
from keras.utils import plot_model

plot_model(model, to_file="Ch16_1.png", show_shapes=True)
