import sounddevice as sd
from scipy.io.wavfile import write

sampleRate = 44100
seconds = 5

myRecording = sd.rec(int(seconds*sampleRate), samplerate=sampleRate, channels=1)
sd.wait()
write('output.wav', sampleRate, myRecording)

