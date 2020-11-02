from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# 載入 IMDb 資料集, 如果是第一次載入會自行下載資料集
top_words = 1000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(
                                num_words=top_words)
max_words = 100
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)

