import os
from bs4 import BeautifulSoup
import re

path = "NICT_JLE_4.1/LightDataSet"
# path = "NICT_JLE_4.1/LearnerOriginal"

output_path = "../../preprocessed_text/"

os.chdir(path)

for input_file in os.listdir():
    if input_file.endswith(".txt"):

        input_text_open = open(input_file,'r')
        input_text = input_text_open.read()

        soup = BeautifulSoup(input_text,'lxml')
        input_text = soup.find_all('b')

        input_string = "".join(str(input_text))

        soup = BeautifulSoup(input_string,'lxml')
        input_text = soup.get_text()

        input_string = "".join(str(input_text))

        input_string = re.sub(">","",input_string)
        input_string = re.sub(">","",input_string)

        pattern = re.compile('[a-zA-Z_ ]')
        matches = pattern.findall(input_string)

        preproccesed_text = "".join(matches)

        os.chdir(output_path)

        output_file = open(input_file, "w") 
        output_file.write(preproccesed_text)
        output_file.close() 

        os.chdir("../")
        os.chdir(path)
