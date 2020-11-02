from keras.preprocessing.text import text_to_word_sequence
# 定義文件
doc = "This is a book. That is a pen."

words = set(text_to_word_sequence(doc))
vocab_size = len(words)
print(vocab_size)

