import os
from collections import Counter
from keras.preprocessing.text import Tokenizer

path = "preprocessed_text"

#PARAMETERS
MIN_OCCURANE = 2
output_name = "vocab"


##################################
#   VOCABULARY
##################################
os.chdir(path)

vocab = Counter()
docs = list()

for input_file in os.listdir():
    if input_file.endswith(".txt"):
        input_text = open(input_file,'r')
        input_string = input_text.read()
        input_text.close()

        tokens = input_string.split()
        vocab.update(tokens)

        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        docs.append(line)


# print("The 50 most common words: "+str(vocab.most_common(50)))

print(docs)

tokens = [k for k,c in vocab.items() if c >= MIN_OCCURANE]

data = ' '.join(tokens)
output_file = open("../"+output_name+".txt",'w')
output_file.write(data)
output_file.close()

##################################
#   DATASET
##################################

tokenizer = Tokenizer()
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtrain.shape)