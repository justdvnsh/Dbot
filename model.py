# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:57:06 2019

@author: Divyansh
"""

from data_pre_processing import sorted_clean_answers, sorted_clean_questions, word2count, answers_into_ints, answersints2words, answerswords2int, questions_into_ints, questionswords2int
import tensorflow as tf
import numpy as np

# Creating placeholders for different inputs.
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob

# Preprocessing the targets, that is adding unique integer for <SOS> token in each batch
# Since models need to process the data in a certain number of batches, and not line by line.
def preprocess_targets(targets, words2int, batch_size):
    # tf.fill will make a matrix of a certain dimention, given by [batch_size and 1] i.e. batch_size rows and 1 column
    # the second arg is the value we are filling it with.
    left_side = tf.fill([batch_size, 1], words2int['<SOS>'])
    # Strided slice slices a certain part the tf tensor like slice slices the part of list in python
    # first arg - targets from which the slice has to be made
    # second arg - beginning of the slice , from where the slice would start
    # third arg - end of the slice , till where the slice has to be made.
    # fourth arg - slide of the slice, how much the slice would slide after each iteration, that is in this case it will move 
    # every [1,1] dimention.
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    # Now we will ust concat the two sides together.
    # First arg - list of the tensors to be concatenated together.
    # second arg - axis along which the tensors woul be concatenated.
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets



