# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 20:16:26 2019

@author: Divyansh
"""

import numpy as np
import pandas as pd
import re 
import time

lines = open('cornell movie-dialogs corpus/movie_lines.txt', encoding = 'utf-8', errors='ignore').read().split('\n')
conversations = open('cornell movie-dialogs corpus/movie_conversations.txt', encoding = 'utf-8', errors='ignore').read().split('\n')

# A Dictionary that maps each line and its id
id2lines = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    ## We only keep the lines that have the lenght of 5 elements , which in this case is almost all the lines.
    if len(_line) == 5:
        id2lines[_line[0]] = _line[4]
        
# A List that maps all the conversations , that is the questions and the answers.
conversation_ids = [] 
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversation_ids.append(_conversation.split(','))
    
# Getting the questions and answers -> The first element in the conversation_ids is the question and the next element is the answer. 
questions = []
answers = [] 
for conversation in conversation_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2lines[conversation[i]])
        answers.append(id2lines[conversation[i+1]])         
        

        
        
        