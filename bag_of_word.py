import os
from collections import Counter
from keras.preprocessing.text import Tokenizer
import numpy as np
from decouple import config
from nltk.corpus import stopwords
from nltk.corpus import words
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import preprocessing
import matplotlib.pyplot as plt

##################################
#   PATHS
##################################

train_data_path = "preprocessed_text/train_data/"
test_data_path = "preprocessed_text/test_data/"

output_train_data_path = "matrix_train_data/"
output_test_data_path = "matrix_test_data/"

##################################
#   PARAMETERS
##################################

min_occurane = config("BOW_MIN_OCCURANE")
min_word_size = config("BOW_MIN_WORD_SIZE")
mode = config("BOW_MODE")
delete_stop_words = config("BOW_DELETE_STOP_WORDS")
# /!\ keep_only_english_words slow computing
keep_only_english_words = config("BOW_KEEP_ONLY_ENGLISH_WORDS")

batch_size = config("BOW_BATCH_SIZE")
epochs = config("BOW_EPOCHS")

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
        if len(k) <= int(min_word_size) or vocab[k] <= int(min_occurane):
            if (delete_stop_words == "true" and k in stop_words) or (keep_only_english_words == "true" and k not in words.words()):
                del vocab[k]
        print("Create vocabulary = "+str(cnt), end="\r")
        cnt+=1

    return vocab

##################################
#   PREPARE DATA
##################################
def prepare_data(train_data_path, test_data_path, vocab) :

    tokenizer = Tokenizer()

    train_dataset = []
    test_dataset = []

    y_train = list()
    y_test = list()

    os.chdir(train_data_path)

    train_data_len = len([name for name in os.listdir() if (os.path.isfile(name) and name.endswith(".txt"))])

    cnt = 0
    for input_file in os.listdir():
        if input_file.endswith(".txt"):
            y_value = [0] * 9
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            y_value[int(input_file.split("_")[1].split(".")[0])-1] = 1
            y_train.append(y_value)

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
            y_value = [0] * 9
            input_text = open(input_file,'r')
            input_string = input_text.read()
            input_text.close()

            y_value[int(input_file.split("_")[1].split(".")[0])-1] = 1
            y_test.append(y_value)

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
    x_train = tokenizer.texts_to_matrix(train_dataset, mode=mode)
    x_test = tokenizer.texts_to_matrix(test_dataset, mode=mode)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test

##################################
#   CLASSIFIER 
##################################
def classifier(input_shape, output_shape):
    model = Sequential()

    model.add(Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

##################################
#   SHOW RESULT
##################################
def show_result(model_history):

    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


##################################
#   PROGRAM
##################################

vocab = create_vocab(train_data_path,test_data_path)

print(vocab.most_common(100))

x_train, y_train, x_test, y_test = prepare_data(train_data_path,test_data_path,vocab)

tf.convert_to_tensor(x_train, dtype=tf.float32)
tf.convert_to_tensor(y_train, dtype=tf.float32)
tf.convert_to_tensor(x_test, dtype=tf.float32)
tf.convert_to_tensor(y_test, dtype=tf.float32)

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
print("x_test :", x_test.shape)
print("y_test : ", y_test.shape)

model = classifier(x_train.shape[1], len(y_train[0]))

history = model.fit(x_train,
                    y_train,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    validation_data=(x_test, y_test),
                    verbose=1)

show_result(history)