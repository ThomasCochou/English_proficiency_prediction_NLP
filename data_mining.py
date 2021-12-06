import os
import pandas as pd
from nltk.corpus import words

# Spell run python embedding.py -t cpu -m uploads/preprocessed_text

##################################
#   GOALS
##################################

# Check numbers in class
# Check mean numbers of english words by class
# Check repetitve

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

##################################
#   ENGLISH WORDS LEN
##################################
def english_word_len(data):
	english_word_len = dict()

	for key in data :
		mean_english_words = 0
		for text in data[key]:
			splitted_data = text.split()
			for word in splitted_data :
				if word in words.words() :
					mean_english_words += 1
		english_word_len[key] = mean_english_words/len(data[key])

	return english_word_len


##################################
#   PROGRAM
##################################

data = split_class(train_data_path,val_data_path)
class_len = class_len(data)
english_word_len = english_word_len(data)

print(pd.DataFrame(class_len, index=["class_len"]))
print(pd.DataFrame(english_word_len, index=["english_word_len"]))

