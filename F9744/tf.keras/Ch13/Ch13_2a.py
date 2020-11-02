import re
from os import listdir
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# IMDb資料所在目錄
path = "/Data/aclImdb/"
# 建立檔案清單
fList = [path + "train/pos/" + x for x in listdir(path + "train/pos")] + \
        [path + "train/neg/" + x for x in listdir(path + "train/neg")] + \
        [path + "test/pos/" + x for x in listdir(path + "test/pos")] + \
        [path + "test/neg/" + x for x in listdir(path + "test/neg")]

# 刪除HTML標籤的符號
def remove_tags(text):
    TAG = re.compile(r'<[^>]+>')
    return TAG.sub('', text)
# 讀取文字檔案的資料    
input_label = ([1] * 12500 + [0] * 12500) * 2
input_text  = []
# 讀取檔案內容
for fname in fList:
    with open(fname, encoding="utf8") as ff:
        input_text += [remove_tags(" ".join(ff.readlines()))]
# 將文件分割成單字        
tok = Tokenizer(num_words=2000)
tok.fit_on_texts(input_text[:25000])
print("文件數: ", tok.document_count)
# 建立訓練和測試資料集
X_train = tok.texts_to_sequences(input_text[:25000])
X_test  = tok.texts_to_sequences(input_text[25000:])
Y_train = input_label[:25000]
Y_test  = input_label[25000:]
# 將序列資料填充成相同長度
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test  = sequence.pad_sequences(X_test,  maxlen=100)
print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
# 定義模型
model = Sequential()
model.add(Embedding(2000, 32, input_length=100))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=["accuracy"])
# 訓練模型
history = model.fit(X_train, Y_train, validation_split=0.2, 
          epochs=5, batch_size=128, verbose=2)
# 評估模型
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
