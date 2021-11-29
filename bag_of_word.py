import os
from collections import Counter
from keras.preprocessing.text import Tokenizer
import numpy as np

train_data_path = "preprocessed_text/train_data/"
test_data_path = "preprocessed_text/test_data/"

output_train_data_path = "matrix_train_data/"
output_test_data_path = "matrix_test_data/"

#PARAMETERS
MIN_OCCURANE = 2
MODE = "binary"


##################################
#   VOCABULARY
##################################
def create_vocab(path, vocab):

    os.chdir(path)

    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            tokens = input_string.split()
            vocab.update(tokens)

    tokens = [k for k,c in vocab.items() if c >= MIN_OCCURANE]

    os.chdir("../../")

    vocab_string = ' '.join(tokens)
    output_file = open("vocab.txt",'w')
    output_file.write(vocab_string)
    output_file.close()

    return tokens, vocab

##################################
#   DATASET
##################################
def transform_dataset(path, tokens, tokenizer) :
    docs = list()

    os.chdir(path)

    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            tokens_dataset = input_string.split()
            tokens_dataset = [w for w in tokens_dataset if w in tokens]
            line = ' '.join(tokens_dataset)
            docs.append(line)

    if path.endswith("train_data/"):
        tokenizer.fit_on_texts(docs)

    tokenized_data = tokenizer.texts_to_matrix(docs, mode=MODE)

    os.chdir("../../")

    return tokenizer, tokenized_data

##################################
#   OUTPUT
##################################

def output_matrix(input_path, output_path, tokenized_data):
    name_list = list()

    os.chdir(input_path)

    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            name_list.append(input_file.split(".")[0])

    if not os.path.exists("../"+output_path):
        os.makedirs("../"+output_path)
    os.chdir("../"+output_path)

    i = 0
    for name in name_list :
        print("Generate matrix = "+str(i)+"/"+str(len(name_list)), end="\r")
        np.savetxt(name+'.out', tokenized_data[i], delimiter=',')
        i += 1

    os.chdir("../../")

##################################
#   PROGRAM
##################################

vocab = Counter()
tokens = []

tokenizer = Tokenizer()

tokens, vocab = create_vocab(train_data_path,vocab)
tokens, vocab = create_vocab(test_data_path,vocab)

print("The 50 most common words: "+str(vocab.most_common(50)))

tokenizer, tokenized_train_data = transform_dataset(train_data_path,tokens,tokenizer)
tokenizer, tokenized_test_data = transform_dataset(test_data_path,tokens,tokenizer)

print("tokenized_train_data.shape : "+str(tokenized_train_data.shape))
print("tokenized_test_data.shape : "+str(tokenized_test_data.shape))

output_matrix(train_data_path, output_train_data_path, tokenized_train_data)
output_matrix(test_data_path, output_test_data_path, tokenized_test_data)
