import os
from bs4 import BeautifulSoup
import re

path = "NICT_JLE_4.1/LightDataSet"
# path = "NICT_JLE_4.1/LearnerOriginal"

output_path = "../../preprocessed_text"

os.chdir(path)


for input_file in os.listdir():
    if input_file.endswith(".txt"):

        input_text_open = open(input_file,'r')
        input_text = input_text_open.read()

        soup = BeautifulSoup(input_text,'lxml')
        input_text = soup.find_all('b')

        preproccesed_text = ""

        for data in input_text :
            data = data.get_text()
            preproccesed_text = preproccesed_text + data + " "

        preproccesed_text = re.sub(">","",preproccesed_text)
        pattern = re.compile('[a-zA-Z_ ]')
        matches = pattern.findall(preproccesed_text)

        preproccesed_text = ""

        for data in matches :
            preproccesed_text = preproccesed_text + data

        os.chdir("../../")
        os.chdir("preprocessed_text/")

        output_file = open(input_file, "w") 
        output_file.write(preproccesed_text)
        output_file.close() 

        os.chdir("../")
        os.chdir(path)
