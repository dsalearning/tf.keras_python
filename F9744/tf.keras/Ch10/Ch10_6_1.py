import numpy as np

samples = ["I hated this movie",
           "This movie is not good"]

token_index = {}

def word_tokenize(text):
    text = text.lower()
    return text.split()
    
for text in samples:
    for word in word_tokenize(text):
        if word not in token_index:
            token_index[word] = len(token_index) + 1
            
print(token_index)            

max_length = 6
results = np.zeros((len(samples), max_length,
                    max(token_index.values())+1 ))

for i, text in enumerate(samples):
    words = list(enumerate(word_tokenize(text)))[:max_length]
    for j, word in words:
        index = token_index.get(word)
        results[i, j, index] = 1.0
        
print(results[0])