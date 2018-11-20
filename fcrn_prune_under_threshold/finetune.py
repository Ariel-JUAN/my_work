#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-13 15:13:15
# @Author  : Arial
# @Link    : http://example.org
# @Version : finetune the pruned model 但是在finetune的时候要mask掉阈值小于0的部分 让他们的梯度不更新 

import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   
import sys
import time
import random
import argparse
import numpy as np
import tensorflow as tf
from datetime import timedelta

from datetime import datetime
from PIL import Image

import models

def readFileNames(path):
    file_names = []
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        else:
            if line[-1]=='\n':
                line = line[:-1]
            file_names.append(line)
    return file_names

def _apply_prune_on_grads(grads_and_vars: list, threshold: float):

    # 得到小于阈值的权重 不更新他们的梯度 对其他权重更新  
    grads_and_vars_sparse = []
    for grad, var in grads_and_vars:
        if 'weights' in var.name:
            small_weights = tf.greater(threshold, tf.abs(var))
            mask = tf.cast(tf.logical_not(small_weights), tf.float32)
            grad = grad * mask

        grads_and_vars_sparse.append((grad, var))
           
    return grads_and_vars_sparse

def shuffle():
    labels_pre = '/home/juanmao/Workspace/dataset/NYU/train_dataset/raw/304*228/depth_all/'
    labels_houzhui = '_depth.png'
    rgb_filelist_temp = [] #rgb_filelist_temp
    depth_filelist_temp = [] #depth_filelist_temp
    Rgb_filelist = []
    Depth_filelist = []
    
    rgb_filelist_temp = readFileNames("/home/juanmao/Workspace/dataset/NYU/train_dataset/nyu_rgb.txt")
    random.shuffle(rgb_filelist_temp)
    for rgb in rgb_filelist_temp:
        labels_path = labels_pre + rgb.split('/')[-1].split('_')[0] + labels_houzhui
        depth_filelist_temp.append(labels_path) 
        
    Rgb_filelist = rgb_filelist_temp#[0:10000]
    Depth_filelist = depth_filelist_temp#[0:10000] 

    return Rgb_filelist, Depth_filelist

def train_one_epoch(sess, Rgb_filelist, Depth_filelist, epoch):
    train_size = num_examples
    train_ptr = 0
    rgbs = []
    labels = []
    total_loss = []
    total_accuracy = []

    for i in range(num_examples//batch_size):   
        # Get next batch of image (path) and labels
        if (train_ptr + batch_size) < train_size:
            rgbs = Rgb_filelist[train_ptr:(train_ptr + batch_size)]
            labels = Depth_filelist[train_ptr:(train_ptr + batch_size)]
            train_ptr += batch_size
        else:
            new_ptr = (train_ptr + batch_size)%train_size
            rgbs = Rgb_filelist[train_ptr :] + Rgb_filelist[ :new_ptr]
            labels = Depth_filelist[train_ptr :] + Depth_filelist[ :new_ptr]
            train_ptr = new_ptr

        # Read images
        rgb_images = np.ndarray([batch_size, height, width, 3])
        label_images = np.ndarray([batch_size, depth_height, depth_width, 1])

        # images as input of network
        for i, rgb in enumerate(rgbs):
            #print rgb
            rgb_img = cv2.imread(rgb)
            rgb_images[i] = rgb_img

        for j, label in enumerate(labels):
            temp_img = cv2.imread(label,-1)
            temp_img = np.array(temp_img).astype('float32')
            temp_img = temp_img/255.0*10
            label_img = np.expand_dims(temp_img, axis=2)
            label_images[j] = label_img

        loss_value, train_, accuracy_= sess.run([loss, train, accuracy], feed_dict={input_node:rgb_images, input_labels:label_images})
        global batches_step
        batches_step += 1
        print('epoch:%d  step:%d  loss:%f rela-accuracy:%f'%(epoch, int(batches_step), loss_value, accuracy_))

        total_loss.append(loss_value)
        total_accuracy.append(accuracy_)

        curve = sess.run(summary, feed_dict={input_node: rgb_images, input_labels: label_images})
        global writer
        writer.add_summary(curve, batches_step)

    saver.save(sess, '/home/juanmao/Workspace/monodepth/fcrn/my_model/NYU.ckpt', global_step=epoch)
    mean_loss = np.mean(total_loss)
    mean_accurancy = np.mean(total_accuracy)
    return mean_loss, mean_accurancy

height = 228
width = 304
depth_height = 128
depth_width = 160
channels = 3
batch_size = 16
# start_learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005
n_epochs = 42

rgb_filelist_temp = readFileNames("/home/juanmao/Workspace/dataset/NYU/train_dataset/nyu_rgb.txt")
depth_filelist_temp = readFileNames("/home/juanmao/Workspace/dataset/NYU/train_dataset/nyu_depth.txt")

Rgb_filelist = []
Depth_filelist = []

rgb_filelist = rgb_filelist_temp
depth_filelist = depth_filelist_temp
num_examples = len(rgb_filelist)

print('The total image number is : ' + str(len(rgb_filelist)))
print('The total image-label number is : ' + str(len(depth_filelist)))

input_node = tf.placeholder(tf.float32, shape=(None, height, width, 3))
input_labels = tf.placeholder(tf.float32, shape=(None, depth_height, depth_width, 1))

# Construct the network
net = models.ResNet50UpProj({'data': input_node}, batch_size, 0.5, True)

# 打印某一个变量 
# restore_var_ = [v for v in tf.global_variables() if 'layer1/weights:0' in v.name]

# finetune最后一层
# restore_var = [v for v in tf.global_variables() if 'ConvPred' not in v.name]
# trainable = [v for v in tf.trainable_variables() if 'ConvPred' in v.name]

restore_var = [v for v in tf.global_variables()]
trainable = [v for v in tf.trainable_variables()]
bn_trainable = [v for v in tf.trainable_variables() if 'bn_conv1' in v.name]
all_trainable = [v for v in tf.trainable_variables() if 'conv' in v.name]+[v for v in tf.trainable_variables() if 'pool' in v.name]+[v for v in tf.trainable_variables() if 'res' in v.name]+[v for v in tf.trainable_variables() if 'Conv' in v.name]
conv_trainable = [v for v in all_trainable if v not in bn_trainable]
save_list = restore_var

#finetune上采样部分 
# not_restore = ['layer', 'ConvPred']
# restore_var = []
# trainable = []
# for each in not_restore:
# 	restore_var += [v for v in tf.global_variables() if each not in v.name]
# 	trainable += [v for v in tf.trainable_variables() if each in v.name]

# finetune上采样部分 
# restore_var = [v for v in tf.global_variables() if 'layer' not in v.name] + [v for v in tf.global_variables() if 'ConvPred' not in v.name]
# trainable = [v for v in tf.trainable_variables() if 'layer' in v.name] + [v for v in tf.trainable_variables() if 'ConvPred' in v.name]

# print(restore_var)
# print('********************************')
# print(trainable)

with tf.name_scope('berhu_loss'):
    labels_mask = tf.cast(tf.not_equal(input_labels, tf.constant(0, dtype=tf.float32)), dtype=tf.float32)
    output = net.get_output()
    output_fill = net.get_output()*labels_mask
    abs_diff = tf.abs(output_fill-input_labels)
    squared_diff = tf.square(abs_diff)
    c = 0.2 * tf.reduce_max(abs_diff)
    diff = tf.where(tf.greater(abs_diff, c), (squared_diff + c*c)/(2.*c), abs_diff)
    sum_diff = tf.reduce_sum(diff)
    loss = sum_diff/tf.reduce_sum(labels_mask) + tf.add_n([weight_decay*tf.nn.l2_loss(v) for v in conv_trainable])
    tf.summary.image('rgb', input_node, max_outputs=3)
    tf.summary.image('output', output, max_outputs=3)
    tf.summary.image('output_fill', output_fill, max_outputs=3)
    tf.summary.image('ground_truth', input_labels, max_outputs=3)
    tf.summary.scalar('loss', loss)

# with tf.name_scope('l1_loss'):
#     labels_mask = tf.cast(tf.not_equal(input_labels, tf.constant(0, dtype=tf.float32)), dtype=tf.float32)
#     output = net.get_output()
#     output_fill = net.get_output()*labels_mask
#     loss = tf.reduce_mean(tf.abs((output-input_labels)*labels_mask))+ tf.add_n([weight_decay*tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#     tf.summary.image('rgb', input_node, max_outputs=3)
#     tf.summary.image('output', output, max_outputs=3)
#     tf.summary.image('output_fill', output_fill, max_outputs=3)
#     tf.summary.image('ground_truth', input_labels, max_outputs=3)
#     tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    where = tf.cast(tf.not_equal(input_labels, tf.constant(0, dtype=tf.float32)), dtype=tf.float32) 
    accuracy = tf.reduce_mean(tf.abs(output*where - input_labels*where)/(input_labels*where+tf.constant(1e-6, dtype=tf.float32)))
    tf.summary.scalar('accuracy', accuracy)

# ckpt train
with tf.name_scope('train'):
    global_step = tf.Variable(0, trainable=False) 
    lr = tf.train.exponential_decay(1e-4, global_step, 5385*4, 0.5, staircase=True)
    train = tf.train.AdamOptimizer(learning_rate=lr)
    grads_and_vars = train.compute_gradients(loss)
    grads_and_vars_sparse = _apply_prune_on_grads(grads_and_vars, threshold)
    train_op = train.apply_gradients(grads_and_vars_sparse, global_step=self.global_step, name='train_op')
    tf.summary.scalar('learning rate', lr)

# npy train
# with tf.name_scope('train'):
#     train = tf.train.MomentumOptimizer(learning_rate, momentum, use_locking=False, name = 'Momentum', use_nesterov=False).minimize(loss)
 
# Build the summary Tensor based on the TF collection of Summaries.
summary = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
# init = tf.global_variables_initializer()
batches_step = 0

config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction = 0.9 
config.gpu_options.allow_growth = True      

sess = tf.Session(config = config)
sess.run(init_op)
writer = tf.summary.FileWriter('/home/juanmao/Workspace/monodepth/fcrn/tensorboard/', sess.graph)
# Saver for storing checkpoints of the model.
saver = tf.train.Saver(var_list=save_list, max_to_keep=None)

print('Start to restore the model')
# model_data_path = '/home/juanmao/Workspace/monodepth/fcrn/model_path/ResNetTensorflow.npy'
# net.load(model_data_path, sess, True) 
# print('The model is restored')

# Use to load from ckpt file
loader = tf.train.Saver(var_list=restore_var)
loader.restore(sess, '/home/juanmao/Workspace/monodepth/fcrn/my_model/NYU.ckpt')
print('The model is restored.')

total_start_time = time.time()
batches_step = 0
for epoch in range(1, n_epochs+1):

    Rgb_filelist, Depth_filelist = shuffle()
    _loss, acc = train_one_epoch(sess, Rgb_filelist, Depth_filelist, epoch)
    print('epoch:%d  mean_loss:%f mean_rela-accuracy:%f'%(epoch, _loss, acc))

total_training_time = time.time() - total_start_time
print('\nTotal training time:%s'%str(timedelta(seconds=total_training_time)))
print('done')

