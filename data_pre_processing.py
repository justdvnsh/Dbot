# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 20:16:26 2019

@author: Divyansh
"""

import numpy as np
import pandas as pd
from data_cleaning import preprocess_sentence

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
clean_questions = [preprocess_sentence(question) for question in questions]
clean_answer = [preprocess_sentence(answer) for answer in answers]

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset():    
    word_pairs = [[question, answer]  for question, answer in zip(clean_questions, clean_answer)]
    return word_pairs[:30000]
	
# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
  def __init__(self, lang):
    self.lang = lang
    self.word2idx = {}
    self.idx2word = {}
    self.vocab = set()
    
    self.create_index()
    
  def create_index(self):
    for phrase in self.lang:
      self.vocab.update(phrase.split(' '))
    
    self.vocab = sorted(self.vocab)
    
    self.word2idx['<pad>'] = 0
    for index, word in enumerate(self.vocab):
      self.word2idx[word] = index + 1
    
    for word, index in self.word2idx.items():
      self.idx2word[index] = word
	  
def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset():
    # creating cleaned input, output pairs
    pairs = create_dataset()

    # index language using the class defined above    
    inp_lang = LanguageIndex(questions for questions, answers in pairs)
    targ_lang = LanguageIndex(answers for questions, answers in pairs)
    
    # Vectorize the input and target languages
    
    # Spanish sentences
    input_tensor = [[inp_lang.word2idx[s] for s in questions.split(' ')] for questions, answers in pairs]
    
    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in answers.split(' ')] for questions, answers in pairs]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    
    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar
	
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset()

from sklearn.model_selection import train_test_split

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 4
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 256
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
