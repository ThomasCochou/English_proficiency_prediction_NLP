import os
from collections import Counter
from keras.preprocessing.text import Tokenizer
import numpy as np
from decouple import config
from nltk.corpus import stopwords
from nltk.corpus import words

train_data_path = "preprocessed_text/train_data/"
test_data_path = "preprocessed_text/test_data/"

output_train_data_path = "matrix_train_data/"
output_test_data_path = "matrix_test_data/"

#PARAMETERS
min_occurane = config("MIN_OCCURANE")
min_word_size = config("MIN_WORD_SIZE")
mode = config("MODE")
delete_stop_words = config("DELETE_STOP_WORDS")
# /!\ keep_only_english_words slow computing
keep_only_english_words = config("KEEP_ONLY_ENGLISH_WORDS")


##################################
#   VOCABULARY
##################################
def create_vocab(path_train,path_test):

    # create Counter and count each words
    vocab = Counter()

    os.chdir(path_train)

    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            tokens = input_string.split()
            vocab.update(tokens)

    os.chdir("../../")

    os.chdir(path_test)

    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            tokens = input_string.split()
            vocab.update(tokens)

    os.chdir("../../")

    stop_words = set(stopwords.words('english'))

    # delete the word from vocab
    # if word size less than min_word_size
    # if word occurane less than min_occurane
    # if word is a stop word from nltk 
    cnt = 0
    for k in list(vocab) :
        if (delete_stop_words == "true" and k in stop_words) or \
        (keep_only_english_words == "true" and k not in words.words()) or \
        len(k) <= int(min_word_size) or \
        vocab[k] <= int(min_occurane):
            del vocab[k]
        print("Create vocabulary = "+str(cnt), end="\r")
        cnt+=1

    return vocab

##################################
#   DATASET
##################################
def transform_dataset(train_data_path, test_data_path, vocab) :

    tokenizer = Tokenizer()

    train_dataset = list()
    test_dataset = list()

    os.chdir(train_data_path)

    train_data_len = len([name for name in os.listdir() if (os.path.isfile(name) and name.endswith(".txt"))])

    cnt = 0
    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            # get only words in vocab
            tokens_dataset = input_string.split()
            tokens_dataset = [w for w in tokens_dataset if w in vocab]
            text = ' '.join(tokens_dataset)
            train_dataset.append(text)
        cnt += 1

    os.chdir("../../")

    os.chdir(test_data_path)

    test_data_len = len([name for name in os.listdir() if (os.path.isfile(name) and name.endswith(".txt"))])

    cnt = 0
    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            # get only words in vocab
            tokens_dataset = input_string.split()
            tokens_dataset = [w for w in tokens_dataset if w in vocab]
            text = ' '.join(tokens_dataset)
            test_dataset.append(text)
        cnt += 1

    os.chdir("../../")

    dataset = train_dataset + test_dataset

    # fit the tokenizer on all the texts
    tokenizer.fit_on_texts(dataset)

    # transform to matrix each texts with differents modes
    tokenized_train_data = tokenizer.texts_to_matrix(train_dataset, mode=mode)
    tokenized_test_data = tokenizer.texts_to_matrix(test_dataset, mode=mode)

    return tokenized_train_data, tokenized_test_data

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
#   Goal : Create a vocabulary, fit dataset to vocabulary and tokenize the text
##################################

vocab = create_vocab(train_data_path,test_data_path)

print(vocab.most_common(100))

tokenized_train_data, tokenized_test_data = transform_dataset(train_data_path,test_data_path,vocab)

print("tokenized_train_data.shape : "+str(tokenized_train_data.shape))
print("tokenized_test_data.shape : "+str(tokenized_test_data.shape))

output_matrix(train_data_path, output_train_data_path, tokenized_train_data)
output_matrix(test_data_path, output_test_data_path, tokenized_test_data)