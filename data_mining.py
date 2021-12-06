import os
import pandas as pd

##################################
#   PATHS
##################################

train_data_path = "preprocessed_text/train_data/"
val_data_path = "preprocessed_text/val_data/"

##################################
#   SPLIT CLASS
##################################
def split_class(path_train,path_val):

	os.chdir(path_train)

	data = {"1" : [],
			"2" : [],
			"3" : [],
			"4" : [],
			"5" : [],
			"6" : [],
			"7" : [],
			"8" : [],
			"9" : []}

	for input_file in os.listdir():
		if input_file.endswith(".txt"):
			text = open(input_file,'r')
			string = text.read()
			data[input_file.split("_")[1].split(".")[0]].append(string);
			text.close()

	os.chdir("../../")

	os.chdir(path_val)

	for input_file in os.listdir():
		if input_file.endswith(".txt"):
			input_text = open(input_file,'r')
			val_string = input_text.read()
			data[input_file.split("_")[1].split(".")[0]].append(string);
			input_text.close()

	os.chdir("../../")

	return data

##################################
#   CLASS LEN
##################################
def class_len(data):
	class_len = dict()

	for key in data :
		class_len[key] = len(data[key])

	return class_len

data = split_class(train_data_path,val_data_path)
class_len = class_len(data)

print(pd.DataFrame(class_len, index=["len"]))