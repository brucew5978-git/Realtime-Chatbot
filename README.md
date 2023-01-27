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



