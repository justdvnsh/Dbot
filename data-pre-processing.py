# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 20:16:26 2019

@author: Divyansh
"""

import numpy as np
import pandas as pd
from data-cleaning import clean_text

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
        
# Cleaning the questions and answers
clean_questions = [clean_text(question) for question in questions]
clean_answer = [clean_text(answer) for answer in answers]
        
# Creating a dictionary that maps each word with its frequency
word2count = {}
for question, answer in zip(clean_questions, clean_answer):
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Creating dictionaries that maps each word of questions and answers with a unique integer
# The Word2count dict contains the words and their frequency , so we will only keep the words which
# appear a certain number of times.
MIN_WORD_FREQUENCY = 20
word_number = 0
questionswords2int = {}
for word, count in word2count.items():
    if count >= MIN_WORD_FREQUENCY:
        questionswords2int[word] = word_number
        word_number += 1
word_number = 0
answerswords2int = {}
for word, count in word2count.items():
    if count >= MIN_WORD_FREQUENCY:
        answerswords2int[word] = word_number
        word_number += 1











        
        