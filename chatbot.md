

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import tensorflow as tf
tf.enable_eager_execution()
import time
import re
import string

# Any results you write to the current directory are saved as output.
```

    ['movie_lines.txt', 'movie_conversations.txt']
    


```python
lines = open('../input/movie_lines.txt', encoding = 'utf-8', errors='ignore').read().split('\n')
conversations = open('../input/movie_conversations.txt', encoding = 'utf-8', errors='ignore').read().split('\n')
```


```python
def preprocess_sentence(w):
    w = w.lower().strip()
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w
```


```python
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

```


```python
# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset():    
    word_pairs = [[question, answer]  for question, answer in zip(clean_questions, clean_answer)]
    return word_pairs[:1000]
```


```python
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
```


```python
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
```


```python
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset()
```


```python
input_tensor
```




    array([[   5,  232, 1715, ...,    0,    0,    0],
           [   5, 1721,    2, ...,    0,    0,    0],
           [   5, 1052, 1559, ...,    0,    0,    0],
           ...,
           [   5,  435, 1518, ...,    0,    0,    0],
           [   5, 1743,    6, ...,    0,    0,    0],
           [   5,  906,    3, ...,    0,    0,    0]], dtype=int32)




```python
input_tensor.shape
```




    (1000, 119)




```python
max_length_inp
```




    119




```python
max_length_targ
```




    147




```python
from sklearn.model_selection import train_test_split

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)
```




    (800, 800, 200, 200)




```python
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.80

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 8
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 256
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
```


```python
def gru(units):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')
```


```python

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
```


```python
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, hidden_size)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))
```


```python
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
```


```python
optimizer = tf.train.AdamOptimizer()


def loss_function(real, pred):
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  return tf.reduce_mean(loss_)
```


```python
import os
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
```


```python
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()
    
    hidden = encoder.initialize_hidden_state()
    total_loss = 0
    
    for (batch, (inp, targ)) in enumerate(dataset):
        loss = 0
        
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, hidden)
            
            dec_hidden = enc_hidden
            
            dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
            
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                
                loss += loss_function(targ[:, t], predictions)
                
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        
        batch_loss = (loss / int(targ.shape[1]))
        
        total_loss += batch_loss
        
        variables = encoder.variables + decoder.variables
        
        gradients = tape.gradient(loss, variables)
        
        optimizer.apply_gradients(zip(gradients, variables))
        
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
```

    Epoch 1 Batch 0 Loss 0.5076
    Epoch 1 Loss 0.4523
    Time taken for 1 epoch 285.9966642856598 sec
    
    Epoch 2 Batch 0 Loss 0.2566
    Epoch 2 Loss 0.4324
    Time taken for 1 epoch 285.82862734794617 sec
    
    Epoch 3 Batch 0 Loss 0.4060
    Epoch 3 Loss 0.4124
    Time taken for 1 epoch 286.3319113254547 sec
    
    Epoch 4 Batch 0 Loss 0.3493
    Epoch 4 Loss 0.3967
    Time taken for 1 epoch 287.2887237071991 sec
    
    Epoch 5 Batch 0 Loss 0.2826
    Epoch 5 Loss 0.3835
    Time taken for 1 epoch 286.78786611557007 sec
    
    Epoch 6 Batch 0 Loss 0.2213
    Epoch 6 Loss 0.3705
    Time taken for 1 epoch 285.91529536247253 sec
    
    Epoch 7 Batch 0 Loss 0.3795
    Epoch 7 Loss 0.3582
    Time taken for 1 epoch 285.344614982605 sec
    
    Epoch 8 Batch 0 Loss 0.2791
    Epoch 8 Loss 0.3469
    Time taken for 1 epoch 284.7535762786865 sec
    
    Epoch 9 Batch 0 Loss 0.2640
    Epoch 9 Loss 0.3351
    Time taken for 1 epoch 284.0793581008911 sec
    
    Epoch 10 Batch 0 Loss 0.3978
    Epoch 10 Loss 0.3233
    Time taken for 1 epoch 283.53520011901855 sec
    
    


```python
def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()

        result += targ_lang.idx2word[predicted_id] + ' '

        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot
```


```python
def answer(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        
    #rint('Input: {}'.format(sentence))
    #rint(': {}'.format(result))
    
    #ttention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    #lot_attention(attention_plot, sentence.split(' '), result.split(' '))
    return result
```


```python
while True:
    question = input("You: ")
    if question.lower() == 'goodbye':
        break
    else:
        result = answer(question, 
                        encoder, 
                        decoder, 
                        inp_lang, 
                        targ_lang, 
                        max_length_inp, 
                        max_length_targ)  
    print('Chatbot: ', result)
```

    You: how are you?
    Chatbot:  it doesn t tell me about back i was it back when it . <end> 
    You: what's your name again ?
    Chatbot:  okay . what just worlds n . . . maybe you know you are eddie . <end> 
    You: okay, so you are eddie
    Chatbot:  oh , eddie . <end> 
    You: hi
    Chatbot:  tamina could know ! i supposed to doin me done , sweetheart ? <end> 
    You: is your name cameron ?
    Chatbot:  i ride for , just for your son , we questioned you wanna have , tabloids chosen for a good . like his get . i used out with me wrong . not s a hell , too . and by your america . <end> 
    You: how do you get your hair to look like that ?
    Chatbot:  again . <end> 
    You: yes
    Chatbot:  does . rights stupid ways run sinned . <end> 
    You: how do you get your hair like that ?
    Chatbot:  william son . i thought <end> 
    You: goodbye
    
