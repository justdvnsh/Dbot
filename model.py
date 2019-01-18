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

# Preprocessing the targets, that is adding unique integer for <SOS> token in each batch and not including the end token <EOS>
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

# Creating the Encoder RNN layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    """
        rnn_inputs = the model inputs , which will be provided by model_input func.
        rnn_size = the size of the rnn input tensors.
        num_layers = number of layers of the cell.
        keep_prob = the dropout probability
        sequence_length = length of list of each question in the batch.
    """
    ## First we make basic lstm cell and then apply basic dropout rate of keep_prob
    ## to increase efficiency.
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    # since the encoder and decoder cells are just long layers of lstm cells , thus we make those layers.
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    # Since the bidirectional rnn returns 2 vars encoder_cell and the state, thus , we use
    # _ to speciy we only want the second output that is encoder_state
    # cell_fw = instance of the rnn cell which looks forward in time , since the rnn we are making is dynamic
    # cell_bw = instance of the rnn cell which looks backward in time, the regular rnn mechanism
    # sequence_length = sequence_length
    # inputs = the model inputs
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32
                                                       )
    return encoder_state
    
    
# Decoding the training set.
def decode_training_set(encoder_state, decoder_cell, decoder_embedding_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):    
    """
        encoder_state = the encoder_state from the encoder cell
        decoder_cell = the decoder cell, the cell in the rnn of the decoder
        decoder_embedding_input = the input on which we will aplly embedding, that is convert each word into unique vector
        sequence_length = sequence length
        decoding_scope = adv. data structure that wraps tensorflow variables.
        output_function = the function used to return the decoder output in the end
        keep_porb = the dropout probability
        batch_size = batch size
    """
    # First we make the attention state, that is a matrix of zeros and then would update weights at the respective areas.
    # we use tf.zeros and then specify the dimentions of the matrix.
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    # attention_keys = the keys that will be compared by the target state
    # attention_values = the values which are used to construct the context , which is fed into the decoder as the first input
    # attention_score_function = used to compute the similarity between the keys and target states
    # attention_construct_function = function used to build the attention states.
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, attention_option= 'bahdanau', num_units = decoder_cell.output_size)
    # now making the traning_decoder_function
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                               attention_keys,
                                                                               attention_values,
                                                                               attention_score_function,
                                                                               attention_construct_function,
                                                                               name = 'attn_dec_train')
    # Since we only want the final decoder output, thus we wil be using the decoder_output from the dynamic_rnn_decoder
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    
    # adding the dropout layer
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
    
# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    """
        encoder_state = the encoder_state from the encoder cell
        decoder_cell = the decoder cell, the cell in the rnn of the decoder
        decoder_embedding_matrix = the part which are embedded, adn we make a matrix of it to pass it to the attention function.
        sos_id = start of string token id
        eos_id = the end of string token id
        maximum_length = the length of teh longest answer one can find in a batch
        num_words = length of all the unique words, ie, length of the answerswords2int dict
        sequence_length = sequence length
        decoding_scope = adv. data structure that wraps tensorflow variables.
        output_function = the function used to return the decoder output in the end
        keep_porb = the dropout probability
        batch_size = batch size
    """
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions

    

