import os
import numpy as np


train_data_matrix_path = "preprocessed_text/matrix_train_data/"
test_data_matrix_path = "preprocessed_text/matrix_test_data/"

def create_input(path):
	os.chdir(path)

	x = list()
	y = list()

	for input_file in os.listdir():
		if input_file.endswith(".out"):
			x.append(np.loadtxt(input_file))
			y.append(input_file.split("_")[1].split(".")[0])

	os.chdir("../../")

	return x,y

x_train, y_train = create_input(train_data_matrix_path)
x_test, y_test = create_input(test_data_matrix_path)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))