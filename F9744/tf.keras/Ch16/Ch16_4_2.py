from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import concatenate

# 定義模型
model_input = Input(shape=(100, 1))
lstm = LSTM(32)(model_input)
# 第 1 個共享特徵提取層的解釋層
extract1 = Dense(16, activation="relu")(lstm)
# 第 2 個共享特徵提取層的解釋層
dense1 = Dense(16, activation="relu")(lstm)
dense2 = Dense(32, activation="relu")(dense1)
extract2 = Dense(16, activation='relu')(dense2)
# 合併 2 個共享特徵提取層的解釋層
merge = concatenate([extract1, extract2])
output = Dense(1, activation="sigmoid")(merge)
model = Model(inputs=model_input, outputs=output)
model.summary()   # 顯示模型摘要資訊
from tensorflow.keras.utils import plot_model

plot_model(model, to_file="Ch16_4_2.png", show_shapes=True)
