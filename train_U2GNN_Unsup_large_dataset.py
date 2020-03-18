#! /usr/bin/env python

import tensorflow as tf
import numpy as np
np.random.seed(123456789)
tf.compat.v1.set_random_seed(123456789)

import os
import time
import datetime
from model_U2GNN_Unsup_multi import U2GNN
import pickle as cPickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *
import statistics
from sklearn.linear_model import LogisticRegression

# Parameters
# ==================================================

parser = ArgumentParser("U2GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="REDDITMULTI5K", help="Name of the dataset.")
parser.add_argument("--embedding_dim", default=4, type=int, help="Dimensionality of character embedding")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=8, type=int, help="Batch Size")
parser.add_argument("--idx_time", default=1, type=int, help="")
parser.add_argument("--num_epochs", default=10, type=int, help="Number of training epochs")
parser.add_argument("--saveStep", default=1, type=int, help="")
parser.add_argument("--allow_soft_placement", default=True, type=bool, help="Allow device soft device placement")
parser.add_argument("--log_device_placement", default=False, type=bool, help="Log placement of ops on devices")
parser.add_argument("--model_name", default='REDDITMULTI5K', help="")
parser.add_argument('--num_sampled', default=32, type=int, help='')
parser.add_argument("--dropout_keep_prob", default=0.5, type=float, help="Dropout keep probability")
parser.add_argument("--num_hidden_layers", default=4, type=int, help="Number of attention layers: the number T of timesteps in Universal Transformer")
parser.add_argument("--num_heads", default=1, type=int, help="Number of attention heads within each attention layer")
parser.add_argument("--ff_hidden_size", default=1024, type=int, help="The hidden size for the feedforward layer")
parser.add_argument("--num_neighbors", default=4, type=int, help="")
parser.add_argument('--degree_as_tag', action="store_false", help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
parser.add_argument('--fold_idx', type=int, default=0, help='the index of fold in 10-fold validation. 0-9.')
parser.add_argument("--iterations", default=8, type=int, help="")
parser.add_argument('--split_idx', type=int, default=0, help='Split data into 10 parts, do U2GAN for each part. It is for large datasets like Reddit')
args = parser.parse_args()
print(args)

# Load data
print("Loading data...")

use_degree_as_tag = False
if args.dataset == 'COLLAB' or args.dataset == 'IMDBBINARY' or args.dataset == 'IMDBMULTI' or\
    args.dataset == 'REDDITBINARY' or args.dataset == 'REDDITMULTI5K':
    use_degree_as_tag = True

large_graphs, num_classes = load_data(args.dataset, use_degree_as_tag)
####split data into 10 parts
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
labels = [graph.label for graph in large_graphs]
idx_list = []
for idx in skf.split(np.zeros(len(labels)), labels):
    idx_list.append(idx[1])
###process each split
graphs = [large_graphs[i] for i in idx_list[args.split_idx]]
graph_labels = np.array([graph.label for graph in graphs])

#
def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])
    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]

    # Adj_block = coo_matrix((Adj_block_elem, (Adj_block_idx_row, Adj_block_idx_cl)), shape=(start_idx[-1], start_idx[-1]))
    return Adj_block_idx_row, Adj_block_idx_cl

Adj_block_idx_row, Adj_block_idx_cl = get_Adj_matrix(graphs)
dict_Adj_block = {}
for i in range(len(Adj_block_idx_row)):
    if Adj_block_idx_row[i] not in dict_Adj_block:
        dict_Adj_block[Adj_block_idx_row[i]] = []
    dict_Adj_block[Adj_block_idx_row[i]].append(Adj_block_idx_cl[i])

def get_graphpool(batch_graph):
    start_idx = [0]
    # compute the padded neighbor list
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1.0] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = np.array(elem)
    idx = np.array(idx)
    idx_row = idx[:,0]
    idx_cl = idx[:,1]

    graph_pool = coo_matrix((elem, (idx_row, idx_cl)), shape=(len(batch_graph), start_idx[-1]))
    return graph_pool
    # return idx_row, idx_cl

graph_pool = get_graphpool(graphs)

# idx_row, idx_cl = preprocess_graphpool(graphs)
# dict_graph_pool = {}
# for i in range(len(idx_row)):
#     if idx_row[i] not in dict_graph_pool:
#         dict_graph_pool[idx_row[i]] = []
#     dict_graph_pool[idx_row[i]].append(idx_cl[i])

X_concat = np.concatenate([graph.node_features for graph in graphs], 0)
print(X_concat.shape)
feature_dim_size = X_concat.shape[1]
vocab_size = X_concat.shape[0]
num_nodes = sum([len(graph.g) for graph in graphs])

class Batch_Loader(object):
    def __init__(self):
        self.sorted_nodes = sorted(dict_Adj_block.keys())

    def __call__(self):
        random_sampled_nodes = np.random.choice(self.sorted_nodes, args.batch_size, replace=False)
        input_x = []
        for input_node in random_sampled_nodes:
            input_x.append([input_node] + list(np.random.choice(dict_Adj_block[input_node], args.num_neighbors, replace=True)))
        input_x = np.array(input_x)
        input_y = input_x[:,0]
        return input_x, np.reshape(input_y, (args.batch_size, 1))

batch_nodes = Batch_Loader()
# x_batch, y_batch = batch_nodes()
# print(x_batch)
# print(y_batch)

print("Loading data... finished!")

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=args.allow_soft_placement, log_device_placement=args.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=session_conf)
    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        u_2_gan = U2GNN(num_hidden_layers=args.num_hidden_layers,
                      vocab_size=vocab_size,
                      hparams_batch_size=args.batch_size,
                      num_sampled=args.num_sampled,
                      initialization=X_concat,
                      feature_dim_size=feature_dim_size,
                      ff_hidden_size=args.ff_hidden_size,
                      seq_length=args.num_neighbors+1
                  )

        # Define Training procedure
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
        grads_and_vars = optimizer.compute_gradients(u_2_gan.total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(args.run_folder, "runs_U2GNN_Unsup_large", args.model_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                u_2_gan.input_x: x_batch,
                u_2_gan.input_y: y_batch,
            }
            _, step, loss = sess.run([train_op, global_step, u_2_gan.total_loss], feed_dict)
            return loss

        num_batches_per_epoch = int((num_nodes - 1) / args.batch_size) + 1
        for epoch in range(1, args.num_epochs+1):
            loss = 0
            for _ in range(num_batches_per_epoch):
                x_batch, y_batch = batch_nodes()
                loss += train_step(x_batch, y_batch)
                # current_step = tf.compat.v1.train.global_step(sess, global_step)
            print(loss)

            # It will give tensor object
            node_embeddings = graph.get_tensor_by_name('W:0')
            node_embeddings = sess.run(node_embeddings)

            # graph_embeddings = []
            # for i in range(len(dict_graph_pool)):
            #     batch_graph_embeddings = node_embeddings[dict_graph_pool[i]]
            #     graph_embed = np.einsum('ij->j', batch_graph_embeddings)
            #     graph_embeddings.append(graph_embed)
            # graph_embeddings = np.reshape(np.concatenate(graph_embeddings, 0), (len(dict_graph_pool), feature_dim_size))

            graph_embeddings = graph_pool.dot(node_embeddings)

            with open(checkpoint_prefix + '-' + str(epoch), 'wb') as f:
                cPickle.dump(graph_embeddings, f)
            print("Save embeddings to {}\n".format(checkpoint_prefix + '-' + str(epoch)))
