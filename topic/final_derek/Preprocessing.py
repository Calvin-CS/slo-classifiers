# Senior Project: Preprocessing Funtion to  remove stop words.
# Author: Derek Fisher, dgf2
# For: CS 396, at Calvin College
# Date May 2019

infile = "../Data/preprocessed_dataset_20100101-20180510-tok.txt"
outfile = "processed_tweets.txt"

# Function removes irrelevant words from file.
delete_list = ["slo_url", "slo_mention", "word_n", "slo_year", "slo_cash", "woodside", "auspol", "adani", "stopadani",
               "ausbiz", "santos", "whitehaven", "tinto", "fortescue", "bhp", "adelaide", "billiton", "csg", "nswpol",
               "nsw", "lng", "don", "rio", "pilliga", "australia", "asx", "just", "today", "great", "says", "like",
               "big", "better", "rite", "would", "SCREEN_NAME", "mining", "former", "qldpod", "qldpol", "qld", "wr",
               "melbourne", "andrew", "fuck", "spadani", "greg", "th", "australians", "http", "https", "rt", "goadani",
               "co", "amp", "riotinto", "carmichael", "abbot", "bill shorten"]
fin = open(infile)
fout = open(outfile, "w+")

wordList = []

# Read lines into a list
file = open(infile, 'rU')
for line in file:
    for word in line.split():
        wordList.append(word)

for word in wordList:
    if len(word) == 3 and word not in delete_list:
        delete_list.append(word)
    elif word.startswith("#") and word not in delete_list:
        delete_list.append(word)

for word in wordList:
    if word in delete_list:
        wordList.remove(word)
    else:
        fout.write(word + " ")

fin.close()
fout.close()
print("preprocessing done")

