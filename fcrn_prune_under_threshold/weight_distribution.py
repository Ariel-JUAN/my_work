import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

reader = tf.train.NewCheckpointReader('/home/juanmao/Workspace/monodepth/fcrn/model_path/NYU_FCRN.ckpt')
print('done')
result = []
variables = reader.get_variable_to_shape_map()
for v in variables:
	weight = reader.get_tensor(v)
	result.extend(list(weight.ravel()))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
ax.hist(result, bins=100, density=0, facecolor='green', edgecolor='black', alpha=0.7, range=(-0.01, 0.01))
ax.set_xlabel('weight value')
ax.set_ylabel('number of weights')
ax.set_title('weight distribution')
fig.savefig('weight.png')

def plot_histogram(weights_list: list,
                   image_name: str,
                   include_zeros=True):

    """A function to plot weights distribution"""

    weights = []
    for w in weights_list:
        weights.extend(list(w.ravel()))

    if not include_zeros:
        weights = [w for w in weights if w != 0]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.hist(weights,
            bins=100,
            facecolor='green',
            edgecolor='black',
            alpha=0.7,
            range=(-0.15, 0.15))

    ax.set_title('Weights distribution')
    ax.set_xlabel('Weights values')
    ax.set_ylabel('Number of weights')

    fig.savefig(image_name + '.png')
