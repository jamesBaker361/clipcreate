#this script is for fidning the most common nouns in the wikiart dataset
import os
from  datasets import load_dataset

data=load_dataset("jlbaker361/wikiart",split="train")

word_dict={}

def remove_punctuation(text):
    # Create a translation table that maps each punctuation character to None
    punctuation_table = str.maketrans('', '', '''!()-[]{};:'"\,<>./?@#$%^&*_~''')
    # Translate the text using the table
    return text.translate(punctuation_table)

for row in data:
    text=row["text"]
    words=text.split()
    for word in words:
        word=remove_punctuation(word)
        if word not in word_dict:
            word_dict[word]=0
        word_dict[word]+=1

sorted_list=sorted([(k,v) for k,v in word_dict.items()],key=lambda x: x[1],reverse=True)[:50]
for thing in sorted_list:
    print(thing)