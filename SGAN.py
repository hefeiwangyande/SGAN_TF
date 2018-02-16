#!/usr/bin/python3
#-*- coding:UTF-8 -*-
'''
。
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

import tensorflow as tf
import numpy as np
import scipy
import scipy.misc
import  shutil
import datetime

from  SGAN_TF.cifar10_data_tensorflow import load_cifar_data
from  SGAN_TF.ops import  deconv2d,l_relu,batch_norm,conv2d

mb_size = 100
out_dir='logs/Sgan_DCGAN'
ITRERATION = 100000

if not os.path.exists(out_dir):
    os.makedirs(out_dir) # make out_dir if it does not exist, copy current script to out_dir
    print ("Created folder {}".format(out_dir))
    shutil.copyfile('SGAN.py', out_dir + '/training_script.py')
    # shutil.copyfile('SGAN_TF.py', 'Tensorboard_SGAN_logs'+ '/training_script.py')
else:
    new_out_dir=out_dir+datetime.datetime.now().strftime('%m%d-%H%M%S')
    os.makedirs(new_out_dir)
    shutil.copyfile('SGAN.py', new_out_dir + '/training_script.py')
    print("folder {} already exists. make a new folder {}.".format(out_dir, new_out_dir))
    out_dir=new_out_dir



pre_enc_weights = np.load('pretrained/encoder.npz')
pre_enc_weights = [pre_enc_weights['arr_{}'.format(k)] for k in range(len(pre_enc_weights.files))]
pre_enc_weights[0] = np.transpose(pre_enc_weights[0], (2,3,1,0))
pre_enc_weights[2] = np.transpose(pre_enc_weights[2], (2,3,1,0))

def encoder(imgs,name):
    with tf.variable_scope('encoder'+name):
        # with argscope(Conv2D, nl=tf.identity, kernel_shape=5, stride=1):
            l = tf.convert_to_tensor(imgs)
            l=tf.nn.relu(conv2d(l,64,kernelload=pre_enc_weights[0],biasload=pre_enc_weights[1],
                     padding='VALID',name='conv1')) #[N,28,28,64]
            print(l.get_shape())
            # [N,14,14,64]
            l = tf.nn.max_pool(l, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
            l=tf.nn.relu(conv2d(l,128,kernelload=pre_enc_weights[2],biasload=pre_enc_weights[3],
                     padding='VALID',name='conv2'))  # [N,10,10,128]
            # [N,5,5,128]
            l = tf.nn.max_pool(l, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
            reshape = tf.reshape(l, [mb_size, -1])
            l_Encoder0=tf.nn.relu(tf.layers.dense(reshape,256,name='E_fc1'))
            l_Encoder1=tf.nn.softmax(tf.layers.dense(l_Encoder0,10,name='E_fc2'))
            return l_Encoder0, l_Encoder1


class Generator:
    '''
                 G0, gen_x = G0(z, h1)
                 imputs:
                 z:[N,Z_DIM]shape
                 h1:encoder output l_Encoder0 shape=[N,256]
                 output:
     '''
    def __init__(self, depths=[1024, 512, 256, 128], s_size=5):
        self.depths = depths + [3]
        self.s_size = s_size
        self.reuse = False

    def __call__(self,z,h1):
        inputs = tf.convert_to_tensor(z)
        h1 = tf.convert_to_tensor(h1)
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs
            l = tf.nn.relu(batch_norm(tf.layers.dense(z, 128,name='G_fc1')))
            l = tf.nn.relu(batch_norm(tf.layers.dense(l, 128, name='G_fc2')))

            l = tf.nn.relu(batch_norm(tf.layers.dense(tf.concat([l, h1], 1), 5 * 5 * 256,name='G_fc3')))  # [N,384][N,256*5*5]-
            l = tf.reshape(l, [-1, self.s_size, self.s_size, self.depths[0]])

            l=tf.nn.relu(batch_norm(deconv2d(l,output_shape=[mb_size,10,10,self.depths[1]],kernelsize=5,stride=2,name='deconv1')))
            l = tf.nn.relu(batch_norm(deconv2d(l, output_shape=[mb_size, 14, 14, self.depths[2]], kernelsize=5, stride=1,padding='VALID',name='deconv2')))
            l = tf.nn.relu(batch_norm(deconv2d(l, output_shape=[mb_size, 28, 28, self.depths[3]], kernelsize=5, stride=2,name='deconv3')))
            l = deconv2d(l, output_shape=[mb_size, 32, 32, self.depths[4]], kernelsize=5, stride=1,padding='VALID',name='deconv4')

            with tf.variable_scope('tanh'):
                outputs = tf.tanh(l, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs

class GaussianNoiseLayer():
    """
    Add random Gaussian noise N(0, sigma^2) of the same shape to img.
    """
    def __init__(self, sigma=0.1):
        """
        Args:
            sigma (float): stddev of the Gaussian distribution.
        """
        self.sigma=sigma
    def _augment(self, img):
        old_dtype = img.dtype
        # img.get_shape()
        noise=tf.random_normal(tf.shape(img),mean=0,stddev=self.sigma,dtype=old_dtype)
        ret = img + noise
        return ret

class Discriminator:
        def __init__(self, depths=[64, 128, 256, 512]):
            self.depths = [3] + depths
            self.reuse = False

        def __call__(self, inputs):
            def leaky_relu(x, leak=0.2):
                return tf.maximum(x, x * leak)

            l = tf.convert_to_tensor(inputs)
            tf.summary.image('D_input', l, max_outputs=10)

            def GlobalAvgPooling(x, data_format='NHWC'):
                assert x.shape.ndims == 4
                assert data_format in ['NHWC', 'NCHW']
                axis = [1, 2] if data_format == 'NHWC' else [2, 3]
                return tf.reduce_mean(x, axis, name='gap')

            with  tf.variable_scope('d', reuse=self.reuse):
                l = GaussianNoiseLayer(sigma=0.05)._augment(l)
                tf.summary.image('D_NoiseLayer', l, max_outputs=10)

                l = leaky_relu((conv2d(l, self.depths[1], kernelsize=3, stride=1, padding='SAME', name='conv1')))
                l = leaky_relu(
                    batch_norm(conv2d(l, self.depths[2], kernelsize=3, stride=2, padding='SAME', name='conv2')))
                l = tf.nn.dropout(l, 0.1, name='dropout1')
                l = leaky_relu(
                    batch_norm(conv2d(l, self.depths[3], kernelsize=3, stride=1, padding='SAME', name='conv3')))
                l = leaky_relu(
                    batch_norm(conv2d(l, self.depths[4], kernelsize=5, stride=2, padding='SAME', name='conv4')))
                l = tf.nn.dropout(l, 0.1, name='dropout2')
                l = leaky_relu(
                    batch_norm(conv2d(l, self.depths[5], kernelsize=3, stride=1, padding='VALID', name='conv5')))
                l_shared = leaky_relu(
                    conv2d(l, self.depths[6], kernelsize=1, stride=1, padding='SAME', name='NIN_conv6'))
                reshape = tf.reshape(l_shared, [mb_size, -1])
                # 重建z_recon,不加激活函数。用于entropy loss.
                z_recon = tf.layers.dense(reshape, 16, name='D_fc2')
                # 输出判断。
                l = GlobalAvgPooling(l_shared, data_format='NHWC')  # [N,192]
                with tf.variable_scope('classify'):
                    outputs = tf.layers.dense(l, 1, name='outputs')
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
            return z_recon, outputs
'load data'
meanimg, data = load_cifar_data('data/')
trainx = data['X_train'] #[50000,3072]
randidx = np.random.randint(len(trainx), size=mb_size)
batch_xs = trainx[randidx]
batch_xs = batch_xs.reshape((mb_size, 32, 32, 3))

z= tf.random_uniform([mb_size, 16], -1, 1, name='z_train')
# g=Generator(depths=[256, 256, 128, 128])
g = Generator(depths=[256, 256, 128, 128])
real_fc1, _ = encoder(batch_xs, name='real')
G_sample= g(z,real_fc1)
recon_fc1, _ = encoder(G_sample, name='fake')

tf.summary.image('generated-samples-cifar',G_sample, max_outputs=30)
# d=Discriminator(depths=[64, 128, 256, 512,512])
d = Discriminator(depths=[96, 96, 192, 192, 192, 192])

_,D_logit_real = d(batch_xs)
z_recon,D_logit_fake= d(G_sample)

with tf.variable_scope('gen_loss'):
    G_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    G_loss_cond= tf.reduce_mean((recon_fc1 - real_fc1) ** 2)
    G_loss_ent = tf.reduce_mean((z_recon - z) ** 2)
    G_loss= G_loss_adv+G_loss_cond+10 * G_loss_ent

with tf.variable_scope('Disc_loss'):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss_adv = D_loss_real + D_loss_fake
    D_loss=D_loss_adv+10 * G_loss_ent

D_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss,var_list=d.variables)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(G_loss,var_list=g.variables)
with tf.control_dependencies([G_solver, D_solver]):
    train_op =tf.no_op(name='train')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 合并到Summary中
merged = tf.summary.merge_all()
# 选定可视化存储目录
writer = tf.summary.FileWriter("Tensorboard_DCGAN_logs/", sess.graph)


num_epoch=ITRERATION/(50000 / mb_size)#200 epoch

for step  in range(int(ITRERATION)):
    _,G_loss_curr, D_loss_curr= sess.run([train_op,G_loss, D_loss])
    G_images= sess.run(G_sample)
    result = sess.run(merged)  # merged也是需要run的
    writer.add_summary(result, step)  # result是summary类型的，需要放入writer中，i步数（x轴）
    print('Iter: {}'.format(step))
    print('D loss: {:.4}'. format(D_loss_curr))
    print('G_loss: {:.4}'.format(G_loss_curr))
    print()

    # for epoch in range(int(num_epoch)):
    if step % 200 == 0:
        ''' original images in the training set'''
        orix = batch_xs
        orix = [orix[i] for i in range(100)]
        rows = []
        for i in range(10):
            rows.append(np.concatenate(orix[i::10], 1))
        orix = np.concatenate(rows, 0)
        scipy.misc.imsave(out_dir + "/cifar_ori_step{}.png".format(step), orix)

        ''' reconstruct images '''
        reconx = G_images
        # reconx = np.reshape(reconx[:100,:,:, ], (100, 3, 32, 32))
        reconx = [reconx[i] for i in range(100)]
        rows = []
        for i in range(10):
            rows.append(np.concatenate(reconx[i::10], 1))
        reconx = np.concatenate(rows, 0)
        scipy.misc.imsave(out_dir + "/cifar_recon_step{}.png".format(step), reconx)

