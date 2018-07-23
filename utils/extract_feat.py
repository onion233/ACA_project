import os
import sys
import librosa
import easydict
import numpy as np

from model import build_model_layer_5


def load_audio(f, sample_rate=22050):
  [audio,sr] = librosa.load(f, dtype='float32', sr=sample_rate, mono=True)
  # import pdb; pdb.set_trace()
  audio = np.tile(audio,3)
  audio *= 256.0
  audio = np.reshape(audio, (1,-1,1))

  return audio

def load(fname):
  # Load the file using std open'''
  f = open(fname,'r')
  data = []
  for line in f.readlines():
    data.append(line.replace('\n','').split(' '))
  f.close()

  return data

def load_weights(SoundNet, model_weights, layers = 5):
  tmp1 = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
  tmp2 = [2, 7, 12, 16, 20]
    
  for i in range(0,layers):
    temp1 = tmp1[i]
    temp2 = tmp2[i]
    weights = model_weights[temp1]['weights'].reshape(SoundNet.layers[temp2].get_weights()[0].shape)
    biases = model_weights[temp1]['biases']
    gamma = model_weights[temp1]['gamma']
    beta = model_weights[temp1]['beta']
    mean = model_weights[temp1]['mean']
    var = model_weights[temp1]['var']
    SoundNet.layers[temp2].set_weights([ weights,biases ])
    SoundNet.layers[temp2+1].set_weights([ gamma, beta, mean,var ])

  # tmp = 'conv8'
  # weights = model_weights[tmp]['weights'].reshape(SoundNet.layers[35].get_weights()[0].shape)
  # biases = model_weights[tmp]['biases']
  # SoundNet.layers[35].set_weights([ weights,biases ])

  # tmp = 'conv8_2'
  # weights = model_weights[tmp]['weights'].reshape(SoundNet.layers[34].get_weights()[0].shape)
  # biases = model_weights[tmp]['biases']
  # SoundNet.layers[34].set_weights([ weights,biases ])

  return SoundNet


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  SoundNet = build_model_layer_5()
  model_weights = np.load('test/sound8.npy', encoding='latin1').item()  
  SoundNet = load_weights(SoundNet,model_weights) 
  result = [] 
 
  ffile = open('filenames.txt')
  # ffile = open('railway.txt')
  while 1:
    audio_file = ffile.readline()
    if not audio_file:
       break
    # import pdb; pdb.set_trace()
    print(audio_file)
    audio  = load_audio(audio_file[0:-1], None)
    feature= SoundNet.predict(audio)
    # print(audio.shape)
    # sys.exit()
    print(feature.shape)
    res = {}
    res['name'] = audio_file[0:-1]
    # res['feature'] = list(feature.reshape(1*8*512,))
    res['feature'] = list(feature.reshape(1*81*256,))
    result.append(res)
#     import pdb; pdb.set_trace()
#     print( sum(sum(sum(feature))) )
    # print( sum(sum(sum(sum(sum(sum(feature)))))) )

  # import pdb; pdb.set_trace()
  import pandas as pd
  tmp=pd.DataFrame.from_records(result)
  tmp.to_csv('res_extract3.csv')
  # import pdb; pdb.set_trace() 
    
    # import pdb; pdb.set_trace()
    # scenes = load('categories/categories_places2.txt')
    # objects = load('categories/categories_imagenet.txt')

    # print(scenes[np.argmax(result[0][0,3,:])])
  # print(objects[np.argmax(result[1][0,0,:])])
