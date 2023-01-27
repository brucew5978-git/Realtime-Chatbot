import matplotlib.pyplot as plt
import torch
import soundfile
import numpy


filename = "output_waveglow.wav"
data, sampleRate = soundfile.read(filename, dtype='float32')
print(data.shape)

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  #waveform = waveform.numpy()
  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)


plot_waveform(data, sampleRate)