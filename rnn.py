from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import re, string
import json
from mlp import getVals, getWordPosInContext, predict_words

FEATURES = ["cq"]
LABEL = "start"
CONTEXT = "context"

def getData(name):
    with open(name) as data_file:    
        data = json.load(data_file)
    
    f = {}
    ids = []
    contexts = []
    context = ""
    for i in range(len(data)):
        for x in range(len(data[i]["paragraphs"])):
            c = data[i]["paragraphs"][x]["context"]
            context += c + " "
            contexts.append(c)
            for y in range(len(data[i]["paragraphs"][x]["qas"])):
                qn = data[i]["paragraphs"][x]["qas"][y]["question"]
                mid = data[i]["paragraphs"][x]["qas"][y]["id"]
                    
                if CONTEXT in f:
                    f[CONTEXT].append(x)
                else:
                    f[CONTEXT] = [x]
                    
                ids.append(mid)

                vals = getVals(c, qn)
                if FEATURES[0] in f:
                    f[FEATURES[0]].append(vals)
                else:
                    f[FEATURES[0]] = [vals]

                if(data[i]["paragraphs"][x]["qas"][y]["answer"] != ""):
                    pos = int(data[i]["paragraphs"][x]["qas"][y]["answer"]["answer_start"])
                    wordPos = getWordPosInContext(pos, c)
                    if LABEL in f:
                        f[LABEL].append(wordPos)
                    else:
                        f[LABEL] = [wordPos]
                        
    return f, contexts, unicode(context).encode('utf8'), ids;

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_punc(lower(s)))

def read_data(content):
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

data, contexts, c, ids = getData('train.json')
training_data = read_data(normalize(c).split(" "))

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

t_data, t_contexts, t_c, t_ids = getData('test.json')
t_training_data = read_data(normalize(t_c).split(" "))
words = predict_words(t_data, t_contexts)
t_dictionary, t_reverse_dictionary = build_dataset(t_training_data)
t_vocab_size = len(t_dictionary)

# Parameters
learning_rate = 0.001
training_iters = 10000
display_step = 1000
n_input = 1

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)


        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
    print("Optimization Finished!")
    
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(t_training_data)-end_offset):
            offset = random.randint(0, n_input+1)


        symbols_in_keys = [ [t_dictionary[ str(t_training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([t_vocab_size], dtype=float)
        symbols_out_onehot[t_dictionary[str(t_training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [t_training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = t_training_data[offset + n_input]
            symbols_out_pred = t_reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
    
    answers = []
    for l in range(len(words)):
        sentence = words[l]
        word = sentence.split()
        if len(word) != n_input:
            continue
        try:
            symbols_in_keys = [t_dictionary[word[i]] for i in range(len(word))]
            for i in range(5):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,t_reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            answers.append(normalize_answer(sentence))
        except:
            print("Word not in dictionary")
            
    data = "Id,Answer\n"
    for i in range(len(t_ids)):
        mid = t_ids[i]
        data += mid + "," + answers[i] + "\n"
    
    data = data.encode('utf-8')
    with open('submission.csv','wb') as file:
        file.write(data)