# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 21:25:35 2019

@author: Divyansh
"""
import re
import string

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
}

def clean_text(text):
    text = text.lower()
    for i in contractions:
        if i in text:
            text = re.sub(i, contractions[i], text)
    punctuations = list(string.punctuation)                
    text = [i for i in text if i not in punctuations]                
    text = "".join(text)
    # joining and splitting have been done 2 times because the places from where the 
    # punctutations were removed , leave an empty space out there, thus we clear it.
    text = text.split()
    return " ".join(text)
