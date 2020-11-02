from keras.preprocessing.text import text_to_word_sequence
# 定義文件
doc = "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,Outcome"
# 將文件分割成單字
words = text_to_word_sequence(doc, lower=False, split=",")
print(words)

