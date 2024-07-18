#use this to find only the top ten classes

import os
from  datasets import load_dataset

data=load_dataset("jlbaker361/wikiart",split="train")

medium_list=["art","painting","drawing"]
subject_list=["man","woman","person"]



def get_count_dict()->dict:
    count_dict={}
    for row in data:
        style=row["style"]
        if style not in count_dict:
            count_dict[style]=0
        count_dict[style]+=1
    return count_dict

def get_word_dict(word_list)->dict:
    word_dict={}
    for row in data:
        style=row["style"]
        if style not in word_dict:
            word_dict[style]=0
        text=row["text"]
        for word in word_list:
            if text.find(word)!=-1:
                word_dict[style]+=1
                break
    return word_dict

count_dict=get_count_dict()

for word_list in [medium_list, subject_list]:
    print(word_list)
    word_dict=get_word_dict(word_list)
    normalized_list=[
        (k,v/count_dict[k]) for k,v in word_dict.items()
    ]
    normalized_list.sort(key=lambda x: x[1],reverse=True)
    for n in normalized_list:
        print("\t",n)
