from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout

# 定義模型
model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), padding="same",
                 input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.summary()   # 顯示模型摘要資訊
# 顯示各神經層
print("神經層數: ", len(model.layers))
for i in range(len(model.layers)):
    print(i, model.layers[i].name)
print("每一層的輸入張量: ")    
for i in range(len(model.layers)):
    print(i, model.layers[i].input)  
print("每一層的輸出張量: ")    
for i in range(len(model.layers)):
    print(i, model.layers[i].output)  