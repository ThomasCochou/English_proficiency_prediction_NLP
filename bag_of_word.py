import os
from collections import Counter
from keras.preprocessing.text import Tokenizer
import numpy as np
from decouple import config
from nltk.corpus import stopwords

train_data_path = "preprocessed_text/train_data/"
test_data_path = "preprocessed_text/test_data/"

output_train_data_path = "matrix_train_data/"
output_test_data_path = "matrix_test_data/"

#PARAMETERS
min_occurane = config("MIN_OCCURANE")
mode = config("MODE")


##################################
#   VOCABULARY
##################################
def create_vocab(path):

    counter = Counter()

    os.chdir(path)

    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            tokens = input_string.split()
            counter.update(tokens)

    words = [k for k,c in counter.items() if c >= int(min_occurane) and len(k) > 1]

    stop_words = set(stopwords.words('english'))
    tokens = [w for w in words if not w in stop_words]

    os.chdir("../../")

    vocab_string = ' '.join(tokens)
    output_file = open("vocab.txt",'w')
    output_file.write(vocab_string)
    output_file.close()

    return tokens

##################################
#   DATASET
##################################
def transform_dataset(path, tokens, tokenizer) :
    docs = list()

    os.chdir(path)

    folder_len = len([name for name in os.listdir() if (os.path.isfile(name) and name.endswith(".txt"))])

    cnt = 0
    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            tokens_dataset = input_string.split()
            tokens_dataset = [w for w in tokens_dataset if w in tokens]
            line = ' '.join(tokens_dataset)
            docs.append(line)
            print("Tokenize = "+str(cnt)+"/"+str(folder_len), end="\r")
        cnt += 1

    if path.endswith("train_data/"):
        tokenizer.fit_on_texts(docs)

    tokenized_data = tokenizer.texts_to_matrix(docs, mode=mode)

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

tokenizer_new = Tokenizer()

tokens_train = create_vocab(train_data_path)
tokens_test = create_vocab(test_data_path)

vocab = list(set(tokens_test+tokens_train))

print(len(vocab))

tokenizer_train, tokenized_train_data = transform_dataset(train_data_path,vocab,tokenizer_new)
tokenizer, tokenized_test_data = transform_dataset(test_data_path,vocab,tokenizer_train)

print("tokenized_train_data.shape : "+str(tokenized_train_data.shape))
print("tokenized_test_data.shape : "+str(tokenized_test_data.shape))

output_matrix(train_data_path, output_train_data_path, tokenized_train_data)
output_matrix(test_data_path, output_test_data_path, tokenized_test_data)
