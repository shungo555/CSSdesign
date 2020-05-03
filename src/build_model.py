#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.models import Model
from keras.layers import Conv2D, Dropout, Add, Multiply, Input, Lambda, Activation, Concatenate
from keras.regularizers import l2
from keras.initializers import TruncatedNormal

from bayer_layers import raw2colormosaic, colormosaic_bilinear, raw2subimages, PixelShuffle, subimages2raw

import numpy as np

def WiG_sub( max_dilation_rate = 4, nb_features=64, Wl2=1E-8, input_shape = (128,128,1), init_stddev=0.01, ouput_channels = 3, Mcc = None, skip_mixed = False ):	
	inp = Input(shape=input_shape)
	x = inp
	
	intp = raw2colormosaic() (x)
	intp = colormosaic_bilinear() (intp)
	
	if( Mcc is not None ):
		w = np.reshape( Mcc.transpose(), (1,1,3,3) )
		intp0 = intp
		intp = Conv2D( 3, (1,1), use_bias = False, weights=[w], trainable=False ) (intp)
	
	x = raw2subimages() (x)
	
	zz = []
	mx = []
	for dilation_rate in range(1,max_dilation_rate+1):
		y = Conv2D( nb_features, (3,3), padding='same', dilation_rate=dilation_rate, kernel_initializer=TruncatedNormal(stddev=init_stddev), kernel_regularizer=l2(Wl2) ) (x)
		m = Conv2D( nb_features, (3,3), padding='same', dilation_rate=dilation_rate, kernel_initializer='zeros', activation='sigmoid' ) (x)
		x = Multiply() ([m,y])
		
		y = x
		for dr in reversed(range(2,dilation_rate)):
			z = Conv2D( nb_features, (3,3), padding='same', dilation_rate=dr, kernel_initializer=TruncatedNormal(stddev=init_stddev), kernel_regularizer=l2(Wl2) ) (y)
			m = Conv2D( nb_features, (3,3), padding='same', dilation_rate=dr, kernel_initializer='zeros', activation='sigmoid' ) (y)
			y = Multiply() ([m,z])
		
		z = Conv2D( ouput_channels*4, (3,3), padding='same', dilation_rate=1, kernel_initializer=TruncatedNormal(stddev=init_stddev), kernel_regularizer=l2(Wl2) ) (y)
		m = Conv2D( ouput_channels*4, (3,3), padding='same', dilation_rate=1, kernel_initializer='zeros', activation='sigmoid' ) (y)
		z = Multiply() ([m,z])
		zz.append(z)
		
		if( skip_mixed ):
			z_ = Conv2D( 4, (3,3), padding='same', dilation_rate=1, kernel_initializer=TruncatedNormal(stddev=init_stddev), kernel_regularizer=l2(Wl2) ) (y)
			m_ = Conv2D( 4, (3,3), padding='same', dilation_rate=1, kernel_initializer='zeros', activation='sigmoid' ) (y)
			m_ = Multiply() ([m_,z_])
			mx.append(m_)
	
	z = Add() (zz)

	z = PixelShuffle(2) (z)
	z = Lambda( lambda x: x[:,1:-1,1:-1,:], lambda s: (s[0], s[1]-2, s[2]-2, s[3] ) ) (z)
	
	if( skip_mixed ):
		m = Add() (mx)
		m = PixelShuffle(2) (m)
		m = Lambda( lambda x: x[:,1:-1,1:-1,:], lambda s: (s[0], s[1]-2, s[2]-2, s[3] ) ) (m)
		m = Activation( 'sigmoid' ) (m)
		
		'''
		ip0 =  Multiply() ([(1-m),intp0])
		ip1 =  Multiply() ([m,intp])
		intp = Add() ([ip0,ip1])
		'''
		intp = Multiply() ([m,intp])
	
	if( ouput_channels > 3 ):
		z0 = Lambda( lambda x: x[:,:,:,:3], lambda s: (s[0], s[1], s[2], 3 ) ) (z)
		z1 = Lambda( lambda x: x[:,:,:,3:], lambda s: (s[0], s[1], s[2], ouput_channels-3 ) ) (z)
		z = Add() ( [intp,z0] )
		z = Concatenate()( [z, z1] )
		
	else:
		z = Add() ( [intp,z] )
	
	return Model(inputs=inp, outputs=z)


def WiG_intp( max_dilation_rate = 4, nb_features=64, Wl2=1E-8, input_shape = (128,128,1), init_stddev=0.01, ouput_channels = 3, Mcc = None ):
	inp = Input(shape=input_shape)
	x = inp
	
	z = raw2colormosaic() (x)
	intp = colormosaic_bilinear() (z)
	
	if( Mcc is not None ):
		w = np.reshape( Mcc.transpose(), (1,1,3,3) )
		intp = Conv2D( 3, (1,1), use_bias = False, weights=[w], trainable=False ) (intp)

	x = intp
	
	zz = [intp]
	for dilation_rate in range(1,max_dilation_rate+1):
		y = Conv2D( nb_features, (3,3), padding='same', dilation_rate=dilation_rate, kernel_initializer=TruncatedNormal(stddev=init_stddev), kernel_regularizer=l2(Wl2) ) (x)
		m = Conv2D( nb_features, (3,3), padding='same', dilation_rate=dilation_rate, kernel_initializer='zeros', activation='sigmoid' ) (x)
		x = Multiply() ([m,y])
		
		y = x
		for dr in reversed(range(2,dilation_rate)):
			z = Conv2D( nb_features, (3,3), padding='same', dilation_rate=dr, kernel_initializer=TruncatedNormal(stddev=init_stddev), kernel_regularizer=l2(Wl2) ) (y)
			m = Conv2D( nb_features, (3,3), padding='same', dilation_rate=dr, kernel_initializer='zeros', activation='sigmoid' ) (y)
			y = Multiply() ([m,z])
		
		z = Conv2D( ouput_channels, (3,3), padding='same', dilation_rate=1, kernel_initializer=TruncatedNormal(stddev=init_stddev), kernel_regularizer=l2(Wl2) ) (y)
		m = Conv2D( ouput_channels, (3,3), padding='same', dilation_rate=1, kernel_initializer='zeros', activation='sigmoid' ) (y)
		z = Multiply() ([m,z])
		
		zz.append(z)
	
	z = Add() (zz)
	
	return Model(inputs=inp, outputs=z)

