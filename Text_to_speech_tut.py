import torch
import torchaudio
import matplotlib.pyplot as plt
import time

import IPython

#Source: https://pytorch.org/tutorials/intermediate/text_to_speech_with_torchaudio.html

torch.random.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else "cpu"


#Text processing

#Character-based encoding
SYMBOLS = '_-!\'(),.:;? abcdefghijklmnopqrstuvwxyz'
lookUp = {s: i for i, s in enumerate(SYMBOLS)}
#lookUp indexes each symbol in array

SYMBOLS = set(SYMBOLS)

def textToSequence(text):
    text = text.lower()
    return [lookUp[s] for s in text if s in SYMBOLS]
#Returns index of each character in target text

processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()
#Pretrained processor returns tensor for input text string
text = "Hello world! Text to speech!"


#Phoneme-based encoding, based on sound of words

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()


#Spectrogram Generation 
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacoTron2 = bundle.get_tacotron2().to(device)

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, _, _ = tacoTron2.infer(processed, lengths)


plt.imshow(spec[0].cpu().detach())
#plt.show()


#Waveform generation
#WaveRNN

waveRNN_time = time.time()

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacoTron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, specLengths, _= tacoTron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, specLengths)

    torchaudio.save("output_waveRNN.wav", waveforms[0:1].cpu(), sample_rate = vocoder.sample_rate)

print("WaveRNN Time: ", time.time() - waveRNN_time)

#Griffin-Lim

GriffinLim_time = time.time()

bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacoTron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, specLengths, _= tacoTron2.infer(processed, lengths)
waveforms, lengths = vocoder(spec, specLengths)

torchaudio.save("output_GriffinLim.wav", waveforms[0:1].cpu(), sample_rate = vocoder.sample_rate)

print("Griffin-Lim: ", time.time()-GriffinLim_time)
'''


#Waveglow
#Waveglow is only the vocoder, still need to proccess the text into a spectrogram
bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacoTron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, specLengths, _= tacoTron2.infer(processed, lengths)


#if torch.cuda.is_available():
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')

'''
'''

print("Using URL")
waveglow = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub",
    "nvidia_waveglow",
    model_math="fp32",
    pretrained=False,
)

checkpoint = torch.hub.load_state_dict_from_url(
    "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_fp32/versions/19.09.0/files/nvidia_waveglowpyt_fp32_20190427",
    progress=False,
    map_location=device,
)

state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

waveglow_time = time.time()

waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to(device)
waveglow.eval()

with torch.no_grad():
    waveforms = waveglow.infer(spec)

torchaudio.save("output_waveglow.wav", waveforms[0:1].cpu(), sample_rate=22050)

print("Waveglow time: ", time.time() - waveglow_time)
