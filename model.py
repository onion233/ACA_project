import os
import sys
import keras

from keras.layers import Input, Conv1D, MaxPooling1D, ZeroPadding1D
from keras.layers import BatchNormalization, Activation
from keras.models import Model

def _add_Zp_Conv(input, zero_padding, conv_filters, conv_kernels, conv_strides):
  # import pdb; pdb.set_trace()  
  output = ZeroPadding1D(padding=zero_padding)(input)
  output = Conv1D(conv_filters, conv_kernels, strides=conv_strides, padding='valid')(output)
  
  return output

def _add_Bn_Relu(input, eps=1e-5):
  output = BatchNormalization(epsilon=eps)(input)
  output = Activation('relu')(output)
  
  return output

def _add_Mp(input, pool_size, pool_strides):
  output = MaxPooling1D(pool_size=pool_size, strides=pool_strides, padding='valid')(input)

  return output

def build_model():
  inp = Input(shape=(None,1))
  
  # layer 1
  x = _add_Zp_Conv(inp, 32, 16, 64, 2)
  x = _add_Bn_Relu(x)
  x = _add_Mp(x, 8, 8) 

  # layer 2
  x = _add_Zp_Conv(x, 16, 32, 32, 2)
  x = _add_Bn_Relu(x)
  x = _add_Mp(x, 8, 8)

  # layer 3
  x = _add_Zp_Conv(x, 8, 64, 16, 2)
  x = _add_Bn_Relu(x)

  # layer 4
  x = _add_Zp_Conv(x, 4, 128, 8, 2)
  x = _add_Bn_Relu(x) 

  # layer 5
  x = _add_Zp_Conv(x, 2, 256, 4, 2)
  x = _add_Bn_Relu(x)
  x = _add_Mp(x, 4, 4)

  # layer 6
  x = _add_Zp_Conv(x, 2, 512, 4, 2)
  x = _add_Bn_Relu(x)

  # layer 7
  x = _add_Zp_Conv(x, 2, 1024, 4, 2)
  x = _add_Bn_Relu(x)
  
  # Object  
  output_Object = _add_Zp_Conv(x, 0, 1000,8, 2)
  # Scene
  output_Scene  = _add_Zp_Conv(x, 0, 401, 8, 2)

  SoundNet = Model(inputs=inp, outputs=[output_Scene, output_Object])
  return SoundNet

if __name__ == '__main__':
  SoundNet = build_model()
  SoundNet.summary()
