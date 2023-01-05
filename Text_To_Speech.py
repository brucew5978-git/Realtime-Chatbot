import time
import torchaudio
import torch

device = 'cuda' if torch.cuda.is_available() else "cpu"
chatbotOutputFile = "chatbot_output_waveRNN.wav"

def waveRNN_TTS(targetText):
    
    waveRNN_time = time.time()

    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

    processor = bundle.get_text_processor()
    tacoTron2 = bundle.get_tacotron2().to(device)
    vocoder = bundle.get_vocoder().to(device)

    with torch.inference_mode():
        processed, lengths = processor(targetText)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, specLengths, _= tacoTron2.infer(processed, lengths)
        waveforms, lengths = vocoder(spec, specLengths)

        torchaudio.save(chatbotOutputFile, waveforms[0:1].cpu(), sample_rate = vocoder.sample_rate)

    return chatbotOutputFile
    #print("WaveRNN Time: ", time.time() - waveRNN_time)

if __name__ == "__main__":
    input = "The square root of nine hundred point zero one is thirty point one"
    waveRNN_TTS(input)