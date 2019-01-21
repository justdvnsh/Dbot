# DBot

My personal chatbot. The website where you can try the chatbot will be released quite soon, as the model is made, where you can talk to the chatbot . 
Don't expect a chatbot as complicated as Assistant or Siri, yet it is a powerful chatbot, you can surely have a good time chatting with. Also, this 
chatbot is trained on movie dialouges , so expect quite filmy dialogues .

# Current Progress

Since I am in need of a personal GPU, I have been training the models on the free GPU sessions on Kaggle and colab. Now , since the free sessions have limits
I could not train the model on massive datasets, but I managed to train the models on 1000 sentences of the [CORNELL-MOVIE-DATASET](https://www.kaggle.com/) dataset,
since it contains both the text and summary. I also tried the model to run on the BBC news dataset, but , unfortunately the free session could not manage the massive size of the dataset.
Thus I trained the model for 10 epochs , using the GRU cell and seq2seq with attention mechanism . So, being said that , these are the results I have got.

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
    
	
# How you can help ?

I am completely estatic , that you have thought of helping me. If you wish to do so, kindly find me and my kernels on KAGGLE, at [https://kaggle.com/justdvnsh](https://kaggle.com/justdvnsh)