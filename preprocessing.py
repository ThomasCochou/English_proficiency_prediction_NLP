import os
from bs4 import BeautifulSoup
import re
import xlrd

path = "NICT_JLE_4.1/LightDataSet"
# path = "NICT_JLE_4.1/LearnerOriginal"

xls_path = "NICT_JLE_4.1/NICT_JLE_list.xls"


#PARAMETERS
output_path = "../../preprocessed_text/"


wb = xlrd.open_workbook(xls_path)
sheet = wb.sheet_by_index(0)

os.chdir(path)


for input_file in os.listdir():
    if input_file.endswith(".txt"):

        input_text = open(input_file,'r')
        input_string = input_text.read()
        input_text.close()

        soup = BeautifulSoup(input_string,'lxml')
        input_soup = soup.find_all('b')
        input_string = "".join(str(input_soup))

        soup = BeautifulSoup(input_string,'lxml')
        input_soup = soup.get_text()
        input_string = "".join(str(input_soup))

        input_string = re.sub(">","",input_string)

        pattern = re.compile('[a-zA-Z_ ]')
        matches = pattern.findall(input_string)
        preproccesed_text = "".join(matches)

        os.chdir(output_path)

        for row_num in range(sheet.nrows):
            row_value = sheet.row_values(row_num)
            if row_value[0].split(".")[0] == input_file.split(".")[0] :
                output_text = open(input_file.split(".")[0]+"_"+str(int(row_value[2]))+".txt", "w") 
                output_text.write(preproccesed_text)
                output_text.close() 

        os.chdir("../"+path)
