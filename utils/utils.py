import os
import sys
import librosa
import easydict
import numpy as np

def load_audio(f, sample_rate=22050):
  [audio,sr] = librosa.load(f, dtype='float32', sr=sample_rate, mono=True)
  # import pdb; pdb.set_trace()
  # audio = np.tile(audio,3)
  audio *= 256.0
  audio = np.reshape(audio, (1,-1,1))

  return audio

def load_label(fname):
  # Load the file using std open'''
  f = open(fname,'r')
  data = []
  for line in f.readlines():
    data.append(line.replace('\n','').split(' '))
  f.close()

  return data



