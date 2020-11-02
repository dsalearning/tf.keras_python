from keras.preprocessing.text import Tokenizer
# 定義 3 份文件
docs = ["Keras is an API designed for human beings, not machines.",
		"Easy to learn and easy to use." ,
		"Keras makes it easy to turn models into products."]
# 建立 Tokenizer
tok = Tokenizer()
# 執行文字資料預處理
tok.fit_on_texts(docs)
# 顯示摘要資訊
print(tok.document_count)
print(tok.word_counts)
print(tok.word_index)
print(tok.word_docs)