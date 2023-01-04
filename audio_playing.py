import sounddevice
import soundfile

filename = "output_waveglow.wav"

data, sampleRate = soundfile.read(filename, dtype='float32')
sounddevice.play(data, sampleRate)
status = sounddevice.wait()