

import numpy as np
import tensorflow as tf

__all__ =['deconv2d','l_relu','batch_norm','conv2d']

def deconv2d(input_, output_shape,
             kernelsize=5,stride=1, stddev=0.02,
             padding="SAME",name="deconv2d"):
    with tf.variable_scope(name, reuse=False):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [kernelsize, kernelsize, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1,stride,stride, 1], padding=padding)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv

def batch_norm(input):
     return tf.layers.batch_normalization(input,training=True)

def l_relu(_x,alpha=0.2 ):

  pos = tf.nn.relu(_x)
  neg = alpha * (_x - abs(_x)) * 0.5
  return pos + neg

def conv2d(x,outchannel,kernelsize=None,kernelload=None,
          biasload=None,w_init_value=0.02,
          b_init_value=0, nl=tf.identity,stride=1,
          padding='SAME',name=None):
    """
    convlutional layer.
    Args:
     x: Tensor, 4D NHWC input maps
     kernelsize = 3
     kernelload,biasload:assign W and b for pretiained
     w_init_value:stddev=0.02
     b_init_value:0.0
    Return:
     conv [N,H,W,C]
    """
    with tf.variable_scope(name, reuse=False):
     #w_init=tf.truncated_normal_initializetf.truncated_normal_initializer(stddev=w_init_value)r(stddev=w_init_value)
     b_init = tf.constant_initializer(b_init_value)

     kernel = tf.get_variable('weights',shape=[kernelsize,kernelsize,x.get_shape()[3],outchannel],initializer=tf.truncated_normal_initializer(stddev=w_init_value), \
                             dtype=tf.float32) if kernelload is None else kernelload


     conv = tf.nn.conv2d(x, kernel,strides=[1,stride,stride,1],padding=padding)

     biases = biasload if biasload is not None else tf.get_variable('biases',outchannel,
                                                                   initializer=b_init,dtype=tf.float32)
     pre_activation = tf.nn.bias_add(conv, biases)
     conv =nl(pre_activation)
     return conv