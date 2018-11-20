import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

threshold = 0.001

def mask_for_big_values(weights, pruning_threshold):
    """A function to get mask"""
    small_weights = np.abs(weights) < pruning_threshold
    return np.logical_not(small_weights)

def get_sparse_values_indices(weights): 
    '''类似切片操作 把weights不等于0的权重从weights取出来'''
    values = weights[weights != 0]
    '''取出对应的索引'''
    indices = np.transpose(np.nonzero(weights))
    return values, indices

def plot_histogram(weights_list: list, image_name: str, include_zeros=True):
    """A function to plot weights distribution"""
    weights = []
    for w in weights_list:
        weights.extend(list(w.ravel()))

    if not include_zeros:
        weights = [w for w in weights if w != 0]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.hist(weights, bins=100, facecolor='green', edgecolor='black', alpha=0.7, range=(-0.01, 0.01))

    ax.set_title('Weights distribution')
    ax.set_xlabel('Weights values')
    ax.set_ylabel('Number of weights')

    fig.savefig(image_name + '.png')

def pruning():
    """A function to prune weights which under the threshold
    这个步骤是用mask直接覆盖小于阈值的值 让他们直接等于0"""
    with tf.Session() as sess: 
        '''加载图'''  
        saver = tf.train.import_meta_graph('/home/juanmao/Workspace/monodepth/fcrn/model_path/NYU_FCRN.ckpt.meta')
        '''加载模型'''
        saver.restore(sess, '/home/juanmao/Workspace/monodepth/fcrn/model_path/NYU_FCRN.ckpt')
        print('The model is restored')
        '''确定要剪枝的部分 得到variable'''
        weights_tf = [v for v in tf.trainable_variables() if 'bn' not in v.name and 'BN' not in v.name]
        '''得到数值''' 
        weights = sess.run(weights_tf) 

        for (weight_matrix, tf_weight_matrix) in zip(weights, weights_tf):
            mask = mask_for_big_values(weight_matrix, threshold)
            sess.run(tf_weight_matrix.assign(weight_matrix * mask))

        weighs_new = sess.run(weights_tf)
        plot_histogram(weighs_new, 'weight_distribution_after_pruning', include_zeros=False)

        saver = tf.train.Saver()
        saver.save(sess, '/home/juanmao/Workspace/monodepth/fcrn/my_model/NYU.ckpt')

def deploy_pruned_model():
    """A function to replace all the matrices in a network with sparse matrices
    其实我觉得和pruning函数实现的功能差不多 
    参考: http://github.com/ex4sperans/pruning_with_tensorflow 中的deploy_pruned_model函数 
    weight_matrices, biases = classifier.sess.run([classifier.weight_matrices,
                                               classifier.biases])
    sparse_layers = []
    # turn dense pruned weights into sparse indices and values
    for weights, bias in zip(weight_matrices, biases):

        values, indices = pruning_utils.get_sparse_values_indices(weights)
        '''确定原来weights的shape'''
        shape = np.array(weights.shape).astype(np.int64) 
        '''用namedtuple存储系列数据'''
        sparse_layers.append(pruning_utils.SparseLayer(values=values.astype(np.float32),
                                                       indices=indices.astype(np.int16),
                                                       dense_shape=shape,
                                                       bias=bias))

    # create sparse classifier
    sparse_classifier = network_sparse.FullyConnectedClassifierSparse(
                                input_size=config_sparse.input_size,
                                n_classes=config_sparse.n_classes,
                                sparse_layers=sparse_layers,
                                model_path=config_sparse.model_path,
                                activation_fn=config_sparse.activation_fn)

    def _build_network(self,
                       inputs: tf.Tensor,
                       sparse_layers: list,
                       activation_fn: callable) -> tf.Tensor:
    
        with tf.variable_scope('network'):
    
            net = inputs
    
            self.weight_tensors = []

            bias_initializer = tf.constant_initializer(0.1)

            for i, layer in enumerate(sparse_layers):
    
                with tf.variable_scope('layer_{layer}'.format(layer=i+1)):

                    # create variables based on sparse values                    
                    with tf.variable_scope('sparse'):

                        indicies = tf.get_variable(name='indicies',
                                                   initializer=layer.indices,
                                                   dtype=tf.int16)

                        values = tf.get_variable(name='values',
                                                 initializer=layer.values,
                                                 dtype=tf.float32)

                        dense_shape = tf.get_variable(name='dense_shape',
                                                      initializer=layer.dense_shape,
                                                      dtype=tf.int64)

                    # create a weight tensor based on the created variables
                    '''用tf.sparse_to_dense得到稀疏矩阵 但是为0的部分还在权重里 并没有切掉'''
                    weights = tf.sparse_to_dense(tf.cast(indicies, tf.int64),
                                                 dense_shape,
                                                 values)

                    self.weight_tensors.append(weights)
        
                    name = 'bias'
                    bias = tf.get_variable(name=name,
                                           initializer=layer.bias)
                    '''用输入和权重相乘的方式 一步步得到网络的输出 但是针对resnet这种multi-branch
                    我觉得不用相乘的方式 直接对矩阵操作 得到稀疏矩阵'''   
                    net = tf.matmul(net, weights) + bias
    
                    if i < len(sparse_layers) - 1:
                        net = activation_fn(net)
    
            return net
    """

if __name__ == '__main__':
    deploy_pruned_model()









	