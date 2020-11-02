import re
from os import listdir
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

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
print(input_text[5])
print(input_label[5])
# 將文件分割成單字, 建立詞索引字典       
tok = Tokenizer(num_words=2000)
tok.fit_on_texts(input_text[:25000])
print("文件數: ", tok.document_count)
print({k: tok.word_index[k] for k in list(tok.word_index)[:10]})
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



