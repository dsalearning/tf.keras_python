from tensorflow.keras.preprocessing.text import Tokenizer
# 定義 3 份文件
docs = ["Keras is an API designed for human beings, not machines.",
		"Easy to learn and easy to use." ,
		"Keras makes it easy to turn models into products."]
# 建立 Tokenizer
tok = Tokenizer()
# 執行文字資料預處理
tok.fit_on_texts(docs)
# 建立序列資料
words = tok.texts_to_sequences(docs)
print(words)
