import json
import tensorflow as tf
import re, string
import collections
import numpy as np
import random

FEATURES = ["cq"]
LABEL = "start"
CONTEXT = "context"

model_root = './models/MLP/'

def getData(name):
    with open(name) as data_file:    
        data = json.load(data_file)
    
    f = {}
    ids = []
    contexts = []
    for i in range(len(data)):
        for x in range(len(data[i]["paragraphs"])):
            c = data[i]["paragraphs"][x]["context"]
            contexts.append(c)
            for y in range(len(data[i]["paragraphs"][x]["qas"])):
                if CONTEXT in f:
                    f[CONTEXT].append(x)
                else:
                    f[CONTEXT] = [x]
                    
                mid = data[i]["paragraphs"][x]["qas"][y]["id"]
                ids.append(mid)
                
                qn = data[i]["paragraphs"][x]["qas"][y]["question"]
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
                        
    return f, ids, contexts;

def input_pos(data_set):
    feature_cols = {k: tf.constant(data_set[k]) for k in FEATURES}
    labels = None
    if LABEL in data_set:
        labels = tf.constant(data_set[LABEL])
    return feature_cols, labels
    
def getModel():
    feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                            hidden_units=[100, 400, 10],
                                            optimizer=tf.train.AdadeltaOptimizer(
                                                    learning_rate=0.1,
                                                    rho=0.001, 
                                                    epsilon=0.01, 
                                                    name='AdaDelta'
                                            ),
                                            model_dir=model_root);
    return regressor
    
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
    """Lower text and remove punctuation and extra whitespace."""
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

def getWordPosInContext(start, context):
    return len(context[:start].split())

def getDict(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def read_data(content):
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

def getVals(context, qn):
    vals = 0
    wDict, rDict = getDict(read_data(normalize(context).split(" ")))
    qns = normalize(qn).split(" ")

    for qn in qns:
        if qn in wDict.keys():
            vals += int(wDict[qn])
            
    return vals

def predict_words(data, contexts):
    model = getModel();
    preds = model.predict(input_fn=lambda:input_pos(data), as_iterable=False);
    words = []
    for i in range(len(preds)):
        start = int(preds[i]);
        con = normalize(contexts[int(data[CONTEXT][i])]).split(" ")
        if len(con) < start:
            start = random.randint(0,start)
        words.append(con[start]);
    return words

def train_words(data):
    model = getModel();
    model.fit(input_fn=lambda:input_pos(data), steps=1000)


data_set, ids, contexts = getData('train.json');
model = getModel();
model.fit(input_fn=lambda:input_pos(data_set), steps=1000)
preds = model.predict(input_fn=lambda:input_pos(data_set), as_iterable=False);

for i in range(len(ids)):
    mid = ids[i]
    start = int(preds[i]);
    con = normalize(contexts[int(data_set[CONTEXT][i])]).split(" ")
    if len(con) < start:
        start = random.randint(0,len(con) - 1)
    ans = con[start];
    print ans