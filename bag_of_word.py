import os
from collections import Counter
from keras.preprocessing.text import Tokenizer

path = "preprocessed_text"

#PARAMETERS
MIN_OCCURANE = 2


##################################
#   VOCABULARY
##################################
os.chdir(path)

vocab = Counter()

for input_file in os.listdir():
    if input_file.endswith(".txt"):
        input_text = open(input_file,'r')
        input_string = input_text.read()
        input_text.close()

        tokens_vocab = input_string.split()
        vocab.update(tokens_vocab)


print("The 50 most common words: "+str(vocab.most_common(50)))

tokens_vocab = [k for k,c in vocab.items() if c >= MIN_OCCURANE]

vocab = ' '.join(tokens_vocab)
output_file = open("../vocab.txt",'w')
output_file.write(vocab)
output_file.close()

##################################
#   DATASET
##################################

docs = list()

for input_file in os.listdir():
    if input_file.endswith(".txt"):
        input_text = open(input_file,'r')
        input_string = input_text.read()
        input_text.close()

        tokens_dataset = input_string.split()
        tokens_dataset = [w for w in tokens_dataset if w in tokens_vocab]
        line = ' '.join(tokens_dataset)
        docs.append(line)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')

print("Xtrain.shape : "+str(Xtrain.shape))

