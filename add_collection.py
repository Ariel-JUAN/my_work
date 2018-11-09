# 参考链接：https://www.cnblogs.com/mr-wuxiansheng/p/6974170.html

import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import time
import random
import argparse
import numpy as np
import tensorflow as tf

with tf.Session() as sess: 
    '''加载图'''  
    saver = tf.train.import_meta_graph('/home/juanmao/Workspace/monodepth/fcrn/model_path/NYU_FCRN.ckpt.meta')
    '''加载模型'''
    saver.restore(sess, '/home/juanmao/Workspace/monodepth/fcrn/model_path/NYU_FCRN.ckpt')
    print('The model is restored')
    
    '''获取所有tensor输入到txt文件'''     
    # ops = [o for o in sess.graph.get_operations()]
    # with open('/home/juanmao/Workspace/monodepth/fcrn/variable.txt', 'a') as f:
       #  for o in ops:
       #  	f.write(str(o.name)+'\n')

    '''得到输入tensor 如果没有重复 需要在后面加:0'''  
    data = sess.graph.get_tensor_by_name('Placeholder:0')
    '''得到输出tensor'''
    output = sess.graph.get_tensor_by_name('ConvPred/ConvPred:0')
    '''加入到集合中''' 
    tf.add_to_collection('input', data)
    tf.add_to_collection('output', output)

    new_saver = tf.train.Saver()
    new_saver.save(sess, '/home/juanmao/Workspace/monodepth/fcrn/my_model/model.ckpt')
