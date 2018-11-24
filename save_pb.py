import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import re
import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(input_checkpoint, output_graph):
    # must point out the origin model output name
    output_node_names = 'ConvPred/BiasAdd'  
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
    # get default graph
    graph = tf.get_default_graph()
    # return a seralized graph as current graph
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # model seralized
        output_graph_def = graph_util.convert_variables_to_constants(sess=sess, input_graph_def=input_graph_def, output_node_names=output_node_names.split(','))

        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph'%len(output_graph_def.node))
        print('done')

def create_nodes_map(graph):
	nodes_map = {}
	for node in graph.node:
		if node.name not in nodes_map.keys():
			nodes_map[node.name] = node
		else:
			raise ValueError('Duplicate node names detected')
	return nodes_map

def _create_adjacency_list(nodes_map, output_node_name):
    adj_list = {}
    already_visited = []
    output_node = nodes_map[output_node_name]
    traversal_queue = [output_node]
    while traversal_queue:
        curr_node = traversal_queue.pop(0)
        curr_node_name = node_name_from_input(curr_node.name)
        print(curr_node_name)
        already_visited.append(curr_node_name)
        for i, input_node_name in enumerate(curr_node.input):
            name = node_name_from_input(input_node_name)
            input_node = nodes_map[name]
            if name not in adj_list:
                adj_list[name] = []
            adj_list[name].append(curr_node_name)
            traversal_queue.append(input_node)
    return adj_list

def node_name_from_input(node_name):
	if node_name.startswith('^'):
		node_name = node_name[1:]
	m = re.search(r"(.*)?:\d+$", node_name)
	if m:
		node_name = m.group(1)
	return node_name

if __name__ == '__main__':
    input_checkpoint = '/home/juanmao/Workspace/monodepth/fcrn/model_path/NYU_FCRN.ckpt'
    output_graph = '/home/juanmao/Workspace/monodepth/fcrn/model_path/NYU_FCRN.pb'
    # freeze_graph(input_checkpoint, output_graph)
    with tf.gfile.GFile(output_graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    nodes_map = create_nodes_map(graph_def)
    if 'Placeholder' in nodes_map:
    	print('biu')
    # print(nodes_map)
    # adj_list = _create_adjacency_list(nodes_map, 'ConvPred/BiasAdd')
    # print(adj_list)


