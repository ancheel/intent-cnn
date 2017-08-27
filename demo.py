#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/Users/lemengwu/PycharmProjects/intend/runs/1502469251/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

datasets = None

# CHANGE THIS: Load data. Load your own data here
dataset_name = cfg["datasets"]["default"]
if FLAGS.eval_train:
    if dataset_name == "mrpolarity":
        datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                             cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    elif dataset_name == "20newsgroup":
        datasets = data_helpers.get_datasets_20newsgroup(subset="test",
                                              categories=cfg["datasets"][dataset_name]["categories"],
                                              shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                              random_state=cfg["datasets"][dataset_name]["random_state"])
    elif dataset_name == "abstract":
        datasets = data_helpers.get_datasets_abstract("data/")
    elif dataset_name == 'intents':
        datasets = data_helpers.get_datasets_intentst("data/")
    x_text, y = data_helpers.load_data_labels(datasets)
    x_raw, y_test = data_helpers.load_data_labels(datasets)
    y_test = np.argmax(y_test, axis=1)
    print("Total number of test examples: {}".format(len(y_test)))
else:
    if dataset_name == "mrpolarity":
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]
    else:
        x_raw = ["Experimental results on a large number of real-world data sets show that the proposed algorithm outperforms existing HMC methods",
                 "In this paper, we overcome these deficiencies by proposing a hierarchy-aware loss function that is more appropriate for HMC."]
        y_test = [2, 1]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print(x_raw)
print(x_test)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        #batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        flag = 1

        while flag == 1:
            X = []
            x_in = input('query:')
            X.append(x_in)
            X.append(x_in)
            x_in_t = np.array(list(vocab_processor.transform(X)))

            x_in_tb= data_helpers.batch_iter(list(x_in_t), FLAGS.batch_size, 1, shuffle=False)

            for x in x_in_tb:
                pred= sess.run(predictions, {input_x: x, dropout_keep_prob: 1.0})

                if pred[0] == 0:
                    print('expert')
                elif pred[0] == 1:
                    print('paper')
                else:
                    print('conference')

            if x_in[0] == 'end':
                flag == 0



