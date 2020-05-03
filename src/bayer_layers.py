#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.engine.topology import Layer
import numpy as np

'''
Bayer: RGGB
'''

class PixelShuffle(Layer):
	def __init__(self, ratio, **kwargs):
		super(PixelShuffle, self).__init__(**kwargs)
		self.ratio = ratio

	def call(self, inputs):
		input_shape = K.int_shape(inputs)
		batch_size, h, w, c = input_shape
		if batch_size is None:
			batch_size = -1

		oh = h * self.ratio
		ow = w * self.ratio
		oc = c // (self.ratio * self.ratio)

		outputs = K.reshape(inputs, (batch_size, h, w, self.ratio, self.ratio, oc))
		outputs = K.permute_dimensions(outputs, (0, 1, 3, 2, 4, 5))
		outputs = K.reshape(outputs, (batch_size, oh, ow, oc))
		return outputs

	def compute_output_shape(self, input_shape):
		return ( input_shape[0], input_shape[1] * self.ratio, input_shape[2] * self.ratio, input_shape[3] // (self.ratio * self.ratio) )

	def get_config(self):
		config = {'ratio': self.ratio }
		base_config = super(PixelShuffle, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class dePixelShuffle(Layer):
	def __init__(self, ratio, **kwargs):
		super(dePixelShuffle, self).__init__(**kwargs)
		self.ratio = ratio

	def call(self, inputs):
		input_shape = K.int_shape(inputs)
		batch_size, h, w, c = input_shape
		if batch_size is None:
			batch_size = -1
			
		oh = h // self.ratio
		ow = w // self.ratio
		oc = c * self.ratio * self.ratio
		
		outputs = K.reshape(inputs, (batch_size, oh, self.ratio, ow, self.ratio, c))
		outputs = K.permute_dimensions(outputs, (0, 1, 3, 2, 4, 5))
		outputs = K.reshape(outputs, (batch_size, oh, ow, oc))
		return outputs	

	def compute_output_shape(self, input_shape):
		return ( input_shape[0], input_shape[1]//self.ratio, input_shape[2]//self.ratio, input_shape[3]*self.ratio*self.ratio )


	def get_config(self):
		config = {'ratio': self.ratio }
		base_config = super(dePixelShuffle, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class raw2subimages(dePixelShuffle):
	def __init__(self, **kwargs):
		super(raw2subimages, self).__init__(ratio=2, **kwargs)

class subimages2raw(PixelShuffle):
	def __init__(self, **kwargs):
		super(subimages2raw, self).__init__(ratio=2, **kwargs)

class raw2colormosaic(Layer):
	def __init__(self, **kwargs):
		super(raw2colormosaic, self).__init__(**kwargs)
		self.mask = None
		self.mask_shape = None

	def get_mask( self, input_shape ):
		if( self.mask_shape is None or self.mask_shape[1] != input_shape[1] or self.mask_shape[2] != input_shape[2] ):
			self.mask_shape = (1, input_shape[1], input_shape[2], 1)
			mask = [np.zeros( self.mask_shape ) for i in range(3) ]

			for row in range(input_shape[1]):
				rr = row % 2
				for col in range(input_shape[2]):
					cc = col % 2
					if( rr == 0 ):
						if( cc == 0 ): # (0,0)
							mask[0][0,row,col,0] = 1
						else:          # (0,1)
							mask[1][0,row,col,0] = 1
					else:
						if( cc == 0 ): # (1,0)
							mask[1][0,row,col,0] = 1
						else:          # (1,1)
							mask[2][0,row,col,0] = 1
			self.mask = [ K.constant( mask[i] ) for i in range(3) ]
			
		return self.mask

	def call(self, inputs):
		input_shape = K.int_shape(inputs)
		mask = self.get_mask( input_shape )
		outputs = [ mask[i] * inputs for i in range(3) ]
		outputs = K.concatenate( outputs, axis=-1 )
		return outputs

	def compute_output_shape(self, input_shape):
		return ( input_shape[0], input_shape[1], input_shape[2], input_shape[3]*3 )

class colormosaic_bilinear(Layer):
	def __init__(self, **kwargs):
		super(colormosaic_bilinear, self).__init__(**kwargs)
		kernel = np.zeros( (3,3, 3,3) )
		
		###
		kernel[0,0, 0,0] = 1/4
		kernel[0,1, 0,0] = 1/2
		kernel[0,2, 0,0] = 1/4

		kernel[1,0, 0,0] = 1/2
		kernel[1,1, 0,0] = 1
		kernel[1,2, 0,0] = 1/2

		kernel[2,0, 0,0] = 1/4
		kernel[2,1, 0,0] = 1/2
		kernel[2,2, 0,0] = 1/4

		###
		kernel[0,0, 1,1] = 0
		kernel[0,1, 1,1] = 1/4
		kernel[0,2, 1,1] = 0

		kernel[1,0, 1,1] = 1/4
		kernel[1,1, 1,1] = 1
		kernel[1,2, 1,1] = 1/4

		kernel[2,0, 1,1] = 0
		kernel[2,1, 1,1] = 1/4
		kernel[2,2, 1,1] = 0

		###
		kernel[0,0, 2,2] = 1/4
		kernel[0,1, 2,2] = 1/2
		kernel[0,2, 2,2] = 1/4

		kernel[1,0, 2,2] = 1/2
		kernel[1,1, 2,2] = 1
		kernel[1,2, 2,2] = 1/2

		kernel[2,0, 2,2] = 1/4
		kernel[2,1, 2,2] = 1/2
		kernel[2,2, 2,2] = 1/4

		self.kernel = K.constant( kernel )

	def call(self, inputs):
		return K.conv2d( inputs, self.kernel )

	def compute_output_shape(self, input_shape):
		return ( input_shape[0], input_shape[1]-2, input_shape[2]-2, input_shape[3] )


if( __name__ == '__main__' ):
	from keras.layers import Input
	from keras.models import Model	

	inp = Input(shape=(32,32,1))
	x = inp
	x = raw2subimages() (x)
	model_sub = Model(inputs=inp, outputs=x)

	x = subimages2raw() (x)
	model_raw = Model(inputs=inp, outputs=x)
	
	raw = np.random.normal( scale = 1.0, size=(1,32,32,1) )
#	raw = np.asarray( list(range(1,32*32+1) ) ).reshape( (1,32,32,1) )
	raw = raw.astype( np.float32 )
	
	pred = model_sub.predict( raw )
	print( raw[0,:,:,0] )
	print()
	for i in range(4):
		print( pred[0,:,:,i] )
	print()


	pred = model_raw.predict( raw )
	print( pred[0,:,:,0] )
	print()

	'''
	inp = Input(shape=(4,4,1))
	
	x = inp
	x = raw2subimages() (x)
	m_raw_sub = Model(inputs=inp, outputs=x)
	
	x = subimages2raw() (x)
	m_raw_sub_raw = Model(inputs=inp, outputs=x)

	raw = np.asarray( list(range(1,17)) )
	raw = raw.reshape( (1,4,4,1) )
	
	print(raw[0,:,:,0])
	print()


	m = m_raw_sub.predict( raw )
	for i in range(4):
		print(m[0,:,:,i])

	m = m_raw_sub_raw.predict( raw )
	print(m[0,:,:,0])

	x = inp
	x = raw2colormosaic() (x)
	m_raw_col = Model(inputs=inp, outputs=x)
	
	m = m_raw_col.predict( raw )
	for i in range(3):
		print(m[0,:,:,i])

	x = colormosaic_bilinear() (x)
	m_raw_col_bilinear = Model(inputs=inp, outputs=x)
	m = m_raw_col_bilinear( K.variable(raw) )
	for i in range(3):
		print(K.get_value(m[0,:,:,i]))
	'''
