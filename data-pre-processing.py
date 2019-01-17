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
        
# Adding special tokens into the dict - <EOS> for end of string , <SOS> for start of string
# <PAD> for user input , <OUT> for filtered out words
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Creating the inverse dictionary of the answerswords2int dict
answersints2words = {w_i: w for w, w_i in answerswords2int.items()}

# Adding the <SOS> and <EOS> at the start and end of each string respectively in the 
# clean_answers list as, this is the target .
for answer in clean_answer:
    answer = '<SOS> ' + answer + ' <EOS>'
    
# Translating all the questions and answers into their respective unique integers.
# and Replacing the words which were filtered out by value of '<OUT>'
questions_into_ints = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_ints.append(ints)
answers_into_ints = []
for answer in clean_answer:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_ints.append(ints)
    
# Sorting questions and answers by their lenghts to optimise the traning process
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 26):
    for i in enumerate(questions_into_ints):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_ints[i[0]])
            sorted_clean_answers.append(answers_into_ints[i[0]])




        
        