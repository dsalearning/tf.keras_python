from keras.preprocessing.text import text_to_word_sequence
# 定義文件
doc = "Keras is an API designed for human beings, not machines."
# 將文件分割成單字
words = text_to_word_sequence(doc)
print(words)

