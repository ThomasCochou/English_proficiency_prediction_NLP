import os
from bs4 import BeautifulSoup
import re

path = "NICT_JLE_4.1/LightDataSet"
# path = "NICT_JLE_4.1/LearnerOriginal"

output_path = "../../preprocessed_text/"

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

        output_text = open(input_file, "w") 
        output_text.write(preproccesed_text)
        output_text.close() 

        os.chdir("../"+path)
