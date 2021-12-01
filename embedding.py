import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from decouple import config
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding,LSTM,GlobalMaxPooling1D,Dense
import numpy as np
import matplotlib.pyplot as plt

train_data_path = "preprocessed_text/train_data/"
test_data_path = "preprocessed_text/test_data/"

output_train_data_path = "matrix_train_data/"
output_test_data_path = "matrix_test_data/"

embedding_max_len_seq = config("EMBEDDING_MAX_LEN_SEQ")
use_glove = config("USE_GLOVE")
batch_size = config("EMBEDDING_BATCH_SIZE")
epochs = config("EMBEDDING_EPOCHS")

##################################
#   LOAD DATA
##################################
def load_data(path_train,path_test):

	os.chdir(path_train)

	x_train = list()
	y_train = list()
	x_test = list()
	y_test = list()

	for input_file in os.listdir():
		if input_file.endswith(".txt"):
			y_value = [0] * 9
			input_text = open(input_file,'r')
			train_string = input_text.read()
			x_train.append(train_string)
			y_value[int(input_file.split("_")[1].split(".")[0])-1] = 1
			y_train.append(y_value)
			input_text.close()

	os.chdir("../../")

	os.chdir(path_test)

	for input_file in os.listdir():
		if input_file.endswith(".txt"):
			y_value = [0] * 9
			input_text = open(input_file,'r')
			test_string = input_text.read()
			x_test.append(test_string)
			y_value[int(input_file.split("_")[1].split(".")[0])-1] = 1
			y_test.append(y_value)
			input_text.close()

	os.chdir("../../")

	return x_train,y_train,x_test,y_test


##################################
#   PREPARE DATA
##################################
def prepare_data(x_train,y_train):

	#Tokenize the sentences
	tokenizer = Tokenizer()

	#preparing vocabulary
	tokenizer.fit_on_texts(list(x_train))

	#converting text into integer sequences
	x_train_seq  = tokenizer.texts_to_sequences(x_train) 
	x_test_seq = tokenizer.texts_to_sequences(x_test)

	#padding to prepare sequences of same length
	x_train_seq  = pad_sequences(x_train_seq, maxlen=int(embedding_max_len_seq))
	x_test_seq = pad_sequences(x_test_seq, maxlen=int(embedding_max_len_seq))

	return x_train_seq,x_test_seq,tokenizer

##################################
#   GloVE EMBEDDING
##################################
def word_embedding(size_of_vocabulary,tokenizer):
	# load the whole embedding into memory
	embeddings_index = dict()
	f = open('glove/glove.6B.300d.txt', encoding="utf8")

	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs

	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))

	# create a weight matrix for words in training docs
	embedding_matrix = np.zeros((size_of_vocabulary, 300))

	for word, i in tokenizer.word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        embedding_matrix[i] = embedding_vector

	return embedding_matrix


##################################
#   CLASSIFIER (without GloVe)
##################################
def classifier(size_of_vocabulary):
	model=Sequential()

	model.add(Embedding(size_of_vocabulary,300,input_length=int(embedding_max_len_seq),trainable=True)) 
	model.add(LSTM(128,return_sequences=True,dropout=0.2))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(64,activation='relu')) 
	model.add(Dense(9,activation='sigmoid')) 

	model.compile(optimizer='adam', loss='binary_crossentropy',metrics=["acc"]) 

	return model

##################################
#   CLASSIFIER (using GloVe)
##################################
def glove_classifier(size_of_vocabulary, embedding_matrix):
	model=Sequential()

	#embedding layer
	model.add(Embedding(size_of_vocabulary,300,weights=[embedding_matrix],input_length=int(embedding_max_len_seq),trainable=False)) 

	#lstm layer
	model.add(LSTM(128,return_sequences=True,dropout=0.2))

	#Global Maxpooling
	model.add(GlobalMaxPooling1D())

	#Dense Layer
	model.add(Dense(64,activation='relu')) 
	model.add(Dense(9,activation='sigmoid')) 

	#Add loss function, metrics, optimizer
	model.compile(optimizer='adam', loss='binary_crossentropy',metrics=["acc"]) 

	return model

##################################
#   SHOW RESULT
##################################
def show_result(model_history):

	acc = model_history.history['acc']
	val_acc = model_history.history['val_acc']
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
x_train,y_train,x_test,y_test = load_data(train_data_path,test_data_path)
x_train_seq,x_test_seq,tokenizer = prepare_data(x_train,y_train)

size_of_vocabulary = len(tokenizer.word_index) + 1 # +1 for padding

print("size of the vocabulary:"+str(len(tokenizer.word_index) + 1))

if use_glove == "true" :
	embedding_matrix = word_embedding(size_of_vocabulary,tokenizer)

	glove_model = glove_classifier(size_of_vocabulary,embedding_matrix)

	glove_model_history = glove_model.fit(np.array(x_train_seq),
		np.array(y_train),
		batch_size=int(batch_size),
		epochs=int(epochs),
		validation_data=(np.array(x_test_seq),np.array(y_test)),
		verbose=1)

	show_result(glove_model_history)

if use_glove == "false" :
	model = classifier(size_of_vocabulary)

	model_history = model.fit(np.array(x_train_seq),
		np.array(y_train),
		batch_size=inv(batch_size),
		epochs=int(epochs),
		validation_data=(np.array(x_test_seq),np.array(y_test)),
		verbose=1)

	show_result(model_history)

else :
	print("choose parameter use_glove")