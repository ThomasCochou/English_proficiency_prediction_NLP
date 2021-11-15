import os
from collections import Counter
from keras.preprocessing.text import Tokenizer

train_data_path = "preprocessed_text/train_data/"
test_data_path = "preprocessed_text/test_data/"

#PARAMETERS
MIN_OCCURANE = 2


##################################
#   VOCABULARY
##################################
def create_vocab(path,vocab):

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
def transform_dataset(path,tokens, tokenizer) :
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

    tokenize_data = tokenizer.texts_to_matrix(docs, mode='freq')

    os.chdir("../../")

    return tokenizer, tokenize_data


##################################
#   PROGRAM
##################################

vocab = Counter()
tokens = []

tokenizer = Tokenizer()

tokens, vocab = create_vocab(train_data_path,vocab)
tokens, vocab = create_vocab(test_data_path,vocab)

print("The 50 most common words: "+str(vocab.most_common(50)))

tokenizer, tokenize_train_data = transform_dataset(train_data_path,tokens,tokenizer)
tokenizer, tokenize_test_data = transform_dataset(test_data_path,tokens,tokenizer)

print("tokenize_train_data.shape : "+str(tokenize_train_data.shape))
print("tokenize_test_data.shape : "+str(tokenize_test_data.shape))