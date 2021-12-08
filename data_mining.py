import os
import pandas as pd
from collections import Counter


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
#   MOST COMMON COUNTER
##################################
def most_commons(data):
	most_commons = dict()

	for key in data :
		counts = Counter()
		for text in data[key]:
			splitted_data = text.split()
			for word in splitted_data :
				counts[word] += 1
		most_commons[key] = counts.most_common(3)

	return most_commons

##################################
#   STUTTERING COUNTER
##################################
def stuttering(data):

	stuttering = dict()

	for key in data :
		count = 0
		stuttering_data = dict()
		for text in data[key]:
			splitted_data = text.split()
			prev_word = "<PREV_WORD>"
			for word in splitted_data :
				if word == prev_word :
						count += 1
				else :
					if prev_word in stuttering_data:
						if stuttering_data[prev_word] < count : 
							stuttering_data[prev_word] = count
					else :
						stuttering_data[prev_word] = count
					prev_word = word
					count = 0
		stuttering[key] = stuttering_data

	for class_key in list(stuttering) :
		for word_key in list(stuttering[class_key]) :
			if stuttering[class_key][word_key] == 0:
				del stuttering[class_key][word_key]
			elif stuttering[class_key][word_key] == 1:
				del stuttering[class_key][word_key]
			elif stuttering[class_key][word_key] == 2:
				del stuttering[class_key][word_key]

	return stuttering

##################################
#   REFLEXION COUNTER
##################################
def reflexion(data):
	reflexion_words = ["err", "er", "um", "uum", "erm"]
	reflexion = dict()

	for key in data :
		counts = Counter()
		text_len = 0
		for text in data[key]:
			text_len = text_len + len(data[key])
			splitted_data = text.split()
			for word in splitted_data :
				if word in reflexion_words :
					counts[word] += 1
		for key_count in counts :
			counts[key_count] = counts[key_count]/text_len

		reflexion[key] = counts

	return reflexion


##################################
#   PROGRAM
##################################

data = split_class(train_data_path,val_data_path)
class_len = class_len(data)
most_commons = most_commons(data)
stuttering = stuttering(data)
reflexion = reflexion(data)

print("\n\nMOST COMMONS \n")
print(most_commons)

print("\n\nCLASS LEN \n")
print(pd.DataFrame(class_len, index=["class_len"]))

print("\n\nSTUTTERING \n")
print(stuttering)

print("\n\nREFLEXION \n")
for key_class in reflexion :
	print(pd.DataFrame(reflexion[key_class], index=[key_class]))
