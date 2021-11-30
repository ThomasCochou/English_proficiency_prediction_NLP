import os
from bs4 import BeautifulSoup
import re
import xlrd
from decouple import config

input_path = "NICT_JLE_4.1/LearnerOriginal"

xls_path = "NICT_JLE_4.1/NICT_JLE_list.xls"


#PARAMETERS
output_path = "preprocessed_text/"

train_data_path = "train_data/"
test_data_path = "test_data/"

ratio_train_test = config("RATIO_TRAIN_TEST")

##################################
#   PROGRAM
#   Goal : Clean the texts, keep words between <b> and lower case it 
##################################


if not os.path.exists(output_path+train_data_path):
	os.makedirs(output_path+train_data_path)

if not os.path.exists(output_path+test_data_path):
	os.makedirs(output_path+test_data_path)


wb = xlrd.open_workbook(xls_path)
sheet = wb.sheet_by_index(0)

os.chdir(input_path)

folder_len = len([name for name in os.listdir() if (os.path.isfile(name) and name.endswith(".txt"))])

cnt = 1

for input_file in os.listdir():
    if input_file.endswith(".txt"):

        # open each file
        input_text = open(input_file,'r', errors='ignore')
        input_string = input_text.read()
        input_text.close()

        # get each string between <b>
        soup = BeautifulSoup(input_string,'lxml')
        input_soup = soup.find_all('b')
        input_string = "".join(str(input_soup))

        # get only text ?
        soup = BeautifulSoup(input_string,'lxml')
        input_soup = soup.get_text()
        input_string = "".join(str(input_soup))

        # delete ">" char
        input_string = re.sub(">","",input_string)

        # get only text and lower case every word
        pattern = re.compile('[a-zA-Z_ ]')
        matches = pattern.findall(input_string.lower())
        preproccesed_text = "".join(matches)

        # split train/test data
        if cnt < folder_len*float(ratio_train_test) :
        	os.chdir("../../"+output_path+train_data_path)
        else :
        	os.chdir("../../"+output_path+test_data_path)

        # save
        for row_num in range(sheet.nrows):
            row_value = sheet.row_values(row_num)
            if row_value[0].split(".")[0] == input_file.split(".")[0] :
                output_text = open(input_file.split(".")[0]+"_"+str(int(row_value[2]))+".txt", "w") 
                output_text.write(preproccesed_text)
                output_text.close() 

        print("Preprocessing = "+str(cnt)+"/"+str(folder_len), end="\r")

        cnt += 1
        os.chdir("../../"+input_path)
