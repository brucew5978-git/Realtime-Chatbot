# Realtime-Chatbot

## Introduction 

Combined NLP, RNNs and time series data to build a chatbot that can respond to natural language input in synthesized speech.

The chat bot was built using sequence-to-sequence model to handle the variable-length input of the user. The model uses RNN encoder network to evaluate historical input data and generates responses using RNN decoder. Luong attention layers are also added in the decoder network to use encoder hidden states for improved response generation. 

To better interact with the user, Silero STT model is used to convert user language input into text to be fed into the chatbot. The waveRNN TTS model is then used to synthesize speech after decoder generates text responses. Being trained on the movie-corpus dataset, the chatbot can generate interesting responses to a variety of user prompts and questions. 

## Files

Chatbot.py: Building and training the seq2seq chatbot model

Speech_To_Text.py: Converts user speech to text using pretrained STT model for chatbot input

Text_To_Speech.py: Synthesizes text from using pretrained TTS model for chatbot output response 

Chatbot_converse.py: Utilizes all three files to run the chatbot program

## Results and Next Steps

The model is able to properly recognize human speech input and generate interesting and understandable responses (examples of chatbot text response shown below). However, due to the chatbot architecture, as well as the speech synthesis process, the turn-around time from user input, to chatbot output takes around 18 seconds which is a significant amount of time. 

<img width="362" alt="image" src="https://user-images.githubusercontent.com/83440706/215196992-c16d70f2-ca09-46d4-9594-e5106ecf9289.png">

In my upcoming project, I plan to address these problems as well as implement a diffusion models for better results.



