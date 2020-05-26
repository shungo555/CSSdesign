from keras import backend as K
K.set_image_data_format('channels_last')

from keras.models import Model
from keras.layers import noise, Lambda, Dropout, Add, Multiply, Input, Activation, Concatenate,Conv2D
from keras.regularizers import Regularizer,l2, l1
from keras.initializers import TruncatedNormal
from keras.constraints import Constraint
import tensorflow as tf
import numpy as np
# import tensorflow.compat.v1 as tf
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.image_tools.cfa import cfa_bayer

class ConstraintMax(Constraint):
    """Constraint Maximum of weights
    """
    def __init__(self, axis=2):
        self.axis = axis
        
    def __call__(self, w):
        w = K.relu(w)
        w /= K.max(w)+K.epsilon()
        return w
    
    def get_config(self):
        return {'axis': self.axis}
    

class ConstraintArea(Constraint):
    """Constraint Maximum of Area (sumation of weights)
    """
    def __init__(self, axis=2):
        self.axis = axis
        
    def __call__(self, w):
        w = K.relu(w)
        w /= K.max(K.sum(w,axis=2))+K.epsilon()
        return w
    
    def get_config(self):
        return {'axis': self.axis}
    

class RegularizerSmooth(Regularizer):
    """Regularize smoothness (second order differential)
    """
    def __init__(self, w_shape, smoothness):
        self.smoothness = smoothness
        self.w_shape = w_shape
        C = np.zeros((self.w_shape, self.w_shape))
        for i in range(self.w_shape):
            if not (i == 0 or i == self.w_shape - 1):
                C[i, i] = -2
                C[i, i - 1] = 1    
                C[i, i + 1] = 1
        self.C = K.constant(C)
        
    def __call__(self, w):        
        regularization = 0
        w = w[0, 0, :, :]
        Cw = K.dot(self.C, w)
        d2w1=K.reshape(K.dot(Cw, K.constant(np.reshape(np.array([1, 0, 0]).T, (3, 1)))), (self.w_shape, 1))
        d2w2=K.reshape(K.dot(Cw, K.constant(np.reshape(np.array([0, 1, 0]).T, (3, 1)))), (self.w_shape, 1))
        d2w3=K.reshape(K.dot(Cw, K.constant(np.reshape(np.array([0, 0, 1]).T, (3, 1)))), (self.w_shape, 1))
        regularization = K.dot(K.transpose(d2w1), d2w1) + K.dot(K.transpose(d2w2), d2w2) + K.dot(K.transpose(d2w3), d2w3)
        return self.smoothness * regularization[0][0]
    

class MyEncoder():
    """encoder class
    -------
        __init__(initial_sensitivity, smoothness, input_shape): initialize function
        my_encoder: main function (define network)
        get_crgb: return cRGB             
    """
    
    def __init__(self, initial_sensitivity, Ls, smoothness = 1E-6, input_shape = (128, 128, 31), trainable=False, wide_color='g'):
        """intialize
        
        Parameters
        ----------
        initial_sensitivity : array (hsi bands, rgb bands, 1, 1)
            initial camera spectral sensitivity
        Ls :  array (hsi bands, hsi bands, 1, 1)
            illuminant
        smoothness : double, optional
            coefficient of smoothness term, by default 1E-2
        input_shape : tuple, optional
            input shape, by default (128, 128, 31)
        """
        self.smoothness = smoothness
        self.input_shape = input_shape
        self.initial_sensitivity = initial_sensitivity
        self.rgb_bandwidth = 3
        self.Ls = Ls
        self.trainable = trainable
        self.wide_color = wide_color
    

    def my_encoder(self):
        """ define encoder network
        Returns
        -------
        self.raw : Model (inputs = [inp, noise, gain_val], outputs = raw)
            raw model
        """
        # input
        inp = Input( shape=self.input_shape, name = 'HSI' )
        
        # iluminant part
        reflect =  Conv2D( self.input_shape[2], (1, 1), padding = 'same', use_bias = False, weights = [self.Ls], name = 'Ls', trainable = False )(inp)
            
        # camera sensitivity part
        crgb = Conv2D( self.rgb_bandwidth, (1, 1), padding = 'same', kernel_regularizer = RegularizerSmooth(self.input_shape[2], self.smoothness), 
                      kernel_constraint = ConstraintMax(), use_bias = False, weights = [self.initial_sensitivity], name = 'CSS', trainable = self.trainable )(reflect)
        
        # CFA pattern
        cfa = cfa_bayer([self.input_shape[0], self.input_shape[1]], wide_color=self.wide_color)
        mask = K.constant(cfa)
        
        # mosaic image
        mosaic = Lambda( lambda x: x * mask, name = 'CFA' )(crgb)
        
        # 1-band mosaic image
        conw = np.ones((1, 1, self.rgb_bandwidth, 1))
        conwb = np.zeros((1,))
        mosaic1 = Conv2D( 1, (1, 1), padding = 'same', activation = 'relu', use_bias = True, weights = [conw, conwb], trainable = False, name = 'Mosaic1band' )(mosaic)
              
        # add noise
        noise = Input(shape = (self.input_shape[0], self.input_shape[1], 1), name = 'Noise')
        noise_mosaic1 = Add(name = 'AddNoise')([noise, mosaic1])
        
        # input and gain
        gain_val = Input(shape = (1,), name = 'InputGain')
        gain = Lambda(lambda x:  K.tile(K.expand_dims(K.expand_dims(x, axis=-1), axis=-1), [self.input_shape[0], self.input_shape[1], 1]) , name = 'AdjustGain')(gain_val)
        # gain3 = Lambda(lambda x:  K.tile(K.expand_dims(K.expand_dims(x, axis=-1), axis=-1), [self.input_shape[0] - 2 * self.YBorder, self.input_shape[1] - 2 * self.YBorder, 3]) , name = 'AdjustGain2')(gain_val)
        # adjust gain and get raw image
        raw = Multiply(name='Raw')([noise_mosaic1, gain])
        self.raw = Model(inputs = [inp, noise, gain_val], outputs = raw)
        
        #############   OPTION   ######################################################################################
        # For middle layer output
        rmask = K.constant(np.reshape(cfa[:, :, :, 0], [cfa.shape[0], cfa.shape[1], cfa.shape[2], 1]))
        gmask = K.constant(np.reshape(cfa[:, :, :, 1], [cfa.shape[0], cfa.shape[1], cfa.shape[2], 1]))
        bmask = K.constant(np.reshape(cfa[:, :, :, 2], [cfa.shape[0], cfa.shape[1], cfa.shape[2], 1]))
        rmosaic = Lambda(lambda x: x * rmask)(raw)
        gmosaic = Lambda(lambda x: x * gmask)(raw)
        bmosaic = Lambda(lambda x: x * bmask)(raw)
        noise_mosaic = Concatenate(axis=-1)([rmosaic, gmosaic, bmosaic])
        
        self.noise_mosaic3 = Model(inputs = [inp, noise, gain_val], outputs =  noise_mosaic)        
        self.gain = Model(inputs = gain_val, outputs =  gain)
        ################################################################################################################
        
        return self.raw 


    def get_noise_mosaic3(self):
        return self.noise_mosaic3
    

    def get_gain(self):
        return self.gain
    
    def get_raw(self):
        return self.raw


class GainLayer():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def gain_define_layer(self):
        gain_val = Input(shape = (1,), name = 'InputGain2')
        gain = Lambda(lambda x:  K.tile(K.expand_dims(K.expand_dims(x, axis=-1), axis=-1), self.input_shape) , name = 'AdjustGain2')(gain_val)
        return  Model(inputs = gain_val, outputs = gain)

    def gain_multi_layer(self):
        inp = Input(shape=self.input_shape, name = 'input' )
        gain = Input(shape = self.input_shape, name = 'InputGain3')
        # out = Multiply(name='Gain')([inp, gain])
        out = Multiply(name='Gain')([inp, gain])
        return Model(inputs = [inp, gain], outputs = out)