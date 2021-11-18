# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019
@author: Reza Azad
Deeplab base model from: https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.optimizers import Adam
import os
import warnings
import numpy as np
import cv2
import keras.backend as K
from keras.models import Model
from keras import layers
from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
import keras
from keras.layers.core import Lambda
from keras.utils.data_utils import get_file
from keras.layers import Add
import cv2

TF_WEIGHTS_PATH = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, l_name = None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.name = l_name
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.upsample_size = conv_utils.normalize_tuple(
                output_size, 2, 'size')
            self.upsampling = None
        else:
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.upsample_size[0]
            width = self.upsample_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True, name = self.name )
        else:
            return K.tf.image.resize_bilinear(inputs, (self.upsample_size[0],
                                                       self.upsample_size[1]),
                                              align_corners=True, name= self.name )

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = layers.Activation('relu')(x)
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation('relu')(x)

    return x

def get_kernel_gussian(kernel_size, Sigma=1, in_channels = 128):
    kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma= Sigma)
    kernel_weights = kernel_weights * kernel_weights.T
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    return kernel_weights
   
def conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return layers.Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        return layers.Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)

def xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                   rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                               kernel_size=1,
                               stride=stride)
        shortcut = layers.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs
   
def mymodel(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, OS=16):
    if not (weights in {'pascal_voc', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `pascal_voc` '
                         '(pre-trained on PASCAL VOC)')

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Deeplabv3+ model is only available with '
                           'the TensorFlow backend.')

    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = layers.Conv2D(32, (3, 3), strides=(2, 2),
               name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    x = layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = layers.Activation('relu')(x)
    
    x = conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x0 = layers.Activation('relu')(x)
    
    x1 = xception_block(x0, [128, 128, 128], 'entry_flow_block1',
                       skip_connection_type='conv', stride=2,
                       depth_activation=False)
    
    x2, skip1 = xception_block(x1, [256, 256, 256], 'entry_flow_block2',
                              skip_connection_type='conv', stride=2,
                              depth_activation=False, return_skip=True)
    
    x = xception_block(x2, [728, 728, 728], 'entry_flow_block3',
                       skip_connection_type='conv', stride=entry_block3_stride,
                       depth_activation=False)
    
    for i in range(16):
        x = xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                           skip_connection_type='sum', stride=1, rate=middle_block_rate,
                           depth_activation=False)
    
    x = xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                       skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                       depth_activation=False)
    
    x = xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                       skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                       depth_activation=True)
    # end of feature extractor
   
    x = layers.Conv2D(1024, (3,3), padding='same', activation='relu', use_bias=False, name='aspp0b4b')(x)
    x = layers.Conv2D(512, (3,3), padding='same', activation='relu', use_bias=False, name='aspp0bb')(x)
    x = layers.Conv2D(256, (3,3), padding='same', use_bias=False, name='aspp0bbbb')(x)
    x = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(x)
    x = layers.Activation('relu', name='aspp0_activation')(x)

    # ## parameters
    kernet_shapes = [3, 5, 7, 9]
    k_value = np.power(2, 1/3)
    sigma   = 1.6
    
    ## Kernel weights for Laplacian pyramid
    Sigma1_kernel = get_kernel_gussian(kernel_size = kernet_shapes[0], Sigma = sigma*np.power(k_value, 1), in_channels = 256)
    Sigma2_kernel = get_kernel_gussian(kernel_size = kernet_shapes[1], Sigma = sigma*np.power(k_value, 2), in_channels = 256)    
    Sigma3_kernel = get_kernel_gussian(kernel_size = kernet_shapes[2], Sigma = sigma*np.power(k_value, 3), in_channels = 256)     
    Sigma4_kernel = get_kernel_gussian(kernel_size = kernet_shapes[3], Sigma = sigma*np.power(k_value, 4), in_channels = 256)        
    
    Sigma1_layer  = layers.DepthwiseConv2D(kernet_shapes[0], use_bias=False, padding='same')
    Sigma2_layer  = layers.DepthwiseConv2D(kernet_shapes[1], use_bias=False, padding='same')
    Sigma3_layer  = layers.DepthwiseConv2D(kernet_shapes[2], use_bias=False, padding='same')
    Sigma4_layer  = layers.DepthwiseConv2D(kernet_shapes[3], use_bias=False, padding='same')

    ## Gussian filtering
    G1 = Sigma1_layer(x)
    G2 = Sigma2_layer(x)
    G3 = Sigma3_layer(x)    
    G4 = Sigma4_layer(x)    
    G0 = x
    ## Laplacian Pyramid
    L0 = G0
    L1 = layers.Subtract()([G0, G1])
    L2 = layers.Subtract()([G1, G2])
    L3 = layers.Subtract()([G2, G3])            
    L4 = layers.Subtract()([G3, G4])  
      
    Dense0 = layers.Dense(256, activation='relu',    kernel_initializer='he_normal', use_bias=False)
    Dense1 = layers.Dense(32 , activation='relu',    kernel_initializer='he_normal', use_bias=False)
    Dense2 = layers.Dense(256, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
    
    L0_1 = layers.Reshape((1, 1, 256))(layers.GlobalAveragePooling2D()(L0))
    L0_1 = Dense2(Dense1(Dense0(L0_1)))

    L1_1 = layers.Reshape((1, 1, 256))(layers.GlobalAveragePooling2D()(L1))
    L1_1 = Dense2(Dense1(Dense0(L1_1)))
    
    L2_1 = layers.Reshape((1, 1, 256))(layers.GlobalAveragePooling2D()(L2))
    L2_1 = Dense2(Dense1(Dense0(L2_1)))
    
    L3_1 = layers.Reshape((1, 1, 256))(layers.GlobalAveragePooling2D()(L3))
    L3_1 = Dense2(Dense1(Dense0(L3_1)))
 
    L4_1 = layers.Reshape((1, 1, 256))(layers.GlobalAveragePooling2D()(L4))
    L4_1 = Dense2(Dense1(Dense0(L4_1)))
    
    m0 = layers.multiply([L0, L0_1])                
    m1 = layers.multiply([L1, L1_1])
    m2 = layers.multiply([L2, L2_1])
    m3 = layers.multiply([L3, L3_1])
    m4 = layers.multiply([L4, L4_1])
    
    m0 = layers.Reshape((16, 16, 1, 256))(m0)
    m1 = layers.Reshape((16, 16, 1, 256))(m1)
    m2 = layers.Reshape((16, 16, 1, 256))(m2)
    m3 = layers.Reshape((16, 16, 1, 256))(m3)
    m4 = layers.Reshape((16, 16, 1, 256))(m4)
    x  = layers.Concatenate(axis=3)([m0, m1, m2, m3, m4])
    x  = layers.Conv3D(256, (1,1,5), activation='relu', use_bias=False, kernel_initializer='he_normal')(x)
    x  = layers.Reshape((16, 16, 256))(x)
    x  = layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x) 
    x  = layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)     
    x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    up1   = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(x))
    mrg1  = layers.concatenate([x2, up1], axis = 3)
    x = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mrg1)
    x = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    up2   = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(x))
    mrg2  = layers.concatenate([x1, up2], axis = 3)
    x = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mrg2)
    x = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    up3   = layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(x))
    mrg3  = layers.concatenate([x0, up3], axis = 3)
    x = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mrg3)
    x = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(x))

    x = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.Conv2D(1, 1, activation = 'sigmoid')(x)   

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = Model(inputs, x, name='new')

    # # load weights

    
    weights_path = get_file('deeplabv3_weights_tf_dim_ordering_tf_kernels.h5',
                            TF_WEIGHTS_PATH, cache_subdir='models')
    model.load_weights(weights_path, by_name=True)
    Sigma1_layer.set_weights([Sigma1_kernel])
    Sigma2_layer.set_weights([Sigma2_kernel])
    Sigma3_layer.set_weights([Sigma3_kernel])
    Sigma4_layer.set_weights([Sigma4_kernel])
    Sigma1_layer.trainable = False
    Sigma2_layer.trainable = False 
    Sigma3_layer.trainable = False
    Sigma4_layer.trainable = False         
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


model = mymodel(input_shape = (256, 256, 3))
#model.summary()                                                 