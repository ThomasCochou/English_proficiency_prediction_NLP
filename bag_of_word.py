import os
from collections import Counter

path = "preprocessed_text"

#PARAMETERS
MIN_OCCURANE = 2
output_name = "vocab"

os.chdir(path)

vocab = Counter()

for input_file in os.listdir():
    if input_file.endswith(".txt"):
        input_text = open(input_file,'r')
        input_string = input_text.read()
        input_text.close()

        tokens = input_string.split()

        vocab.update(tokens)

print("The 50 most common words: "+str(vocab.most_common(50)))


tokens = [k for k,c in vocab.items() if c >= MIN_OCCURANE]

data = ' '.join(tokens)
output_file = open("../"+output_name+".txt",'w')
output_file.write(data)
output_file.close()
