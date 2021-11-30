import os, pathlib, shutil, random
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SpatialDropout1D, Dropout, Conv1D, LSTM
from keras import preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.constraints import maxnorm

train_data_matrix_path = "preprocessed_text/matrix_train_data/"
test_data_matrix_path = "preprocessed_text/matrix_test_data/"


def to_tensor(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

def create_input(path):
    os.chdir(path)

    x = list()
    y = list()

    folder_len = len([name for name in os.listdir() if (os.path.isfile(name) and name.endswith(".out"))])

    cnt = 0
    for input_file in os.listdir():
        if input_file.endswith(".out"):
            y_value = [0] * 9
            x.append(np.loadtxt(input_file))
            y_value[int(input_file.split("_")[1].split(".")[0])-1] = 1
            y.append(y_value)
        print("Classifier = "+str(cnt)+"/"+str(folder_len), end="\r")
        cnt += 1


    os.chdir("../../")

    return x,y


x_train, y_train = create_input(train_data_matrix_path)
x_test, y_test = create_input(test_data_matrix_path)

x_train = to_tensor(x_train)
y_train = to_tensor(y_train)
x_test = to_tensor(x_test)
y_test = to_tensor(y_test)

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
print("x_test :", x_test.shape)
print("y_test : ", y_test.shape)

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=300,
                    batch_size=32,
                    validation_data=(x_test, y_test))

history_dict = history.history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

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

# Définir le modèle
# model = Sequential()
# model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[0]))
# model.add(SpatialDropout1D(0.2))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(9, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit()

# model = Sequential()
# model.add(Dense(50, input_shape=x_train.shape, activation='relu'))
# model.add(Dense(9, activation='softmax'))
# model.summary()

# opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=50, verbose=2)

