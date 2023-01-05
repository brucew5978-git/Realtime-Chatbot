import torch.nn as nn
import torch
from Chatbot import EncoderRNN, LuongAttentionDecoderRNN, GreedySearchDecoder, evaluateInterface, attentionModel, vocabulary
import sounddevice
from scipy.io.wavfile import write

from Speech_To_Text import SpeechToText
from Text_To_Speech import waveRNN_TTS

import soundfile


hiddenSize = 500
embedding = nn.Embedding(vocabulary.num_words, hiddenSize)
encoderNLayers = 2
decoderNLayers = 2
dropout = 0.1

encoder = EncoderRNN(hiddenSize, embedding, encoderNLayers, dropout)
decoder = LuongAttentionDecoderRNN(attentionModel, embedding, hiddenSize, vocabulary.num_words, decoderNLayers,
                                   dropout)

ENCODER_FILE = "models/chatbot_encoder.pth"
DECODER_FILE = "models/chatbot_decoder.pth"

encoder.load_state_dict(torch.load(ENCODER_FILE, map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(DECODER_FILE, map_location=torch.device('cpu')))


# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)



sampleRate = 44100
seconds = 5

userInputWavFile = 'user_output.wav'


userResponse = ''

while(userResponse != "q"):
    
    print("Recieving audio data...")
    #Recording user voice
    myRecording = sounddevice.rec(int(seconds*sampleRate), samplerate=sampleRate, channels=1)
    sounddevice.wait()
    write(userInputWavFile, sampleRate, myRecording)
    print("Audio data recieved!")

    userVoiceInput = SpeechToText(userInputWavFile)

    print("Thinking of response...")
    chatbotResponse = evaluateInterface(encoder, decoder, searcher, vocabulary, userVoiceInput)  

    print("Generating audio...")
    chatbotResponseWavFile = waveRNN_TTS(chatbotResponse)

    data, sampleRate = soundfile.read(chatbotResponseWavFile, dtype='float32')
    sounddevice.play(data, sampleRate)
    status = sounddevice.wait()
    
    print("Enter q to quit program, or c for continue: ")
    userResponse = input("> ")
    
    #if(userResponse == "q"):
    #    break

