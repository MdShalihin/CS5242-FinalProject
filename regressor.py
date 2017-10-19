import json
import tensorflow as tf
import re, string

FEATURES = ["context", "qns"]
LABEL_POS = "pos"
LABEL_LEN = "length"
model_root = './models/MLP/regressor/'
model_pos = 'pos/'
model_len = 'len/'

def getData(name):
    with open(name) as data_file:    
        data = json.load(data_file)
    
    f = {}
    ids = []
    for i in range(len(data)):
        for x in range(len(data[i]["paragraphs"])):
            c = data[i]["paragraphs"][x]["context"]
            for y in range(len(data[i]["paragraphs"][x]["qas"])):
                qn = data[i]["paragraphs"][x]["qas"][y]["question"]
                mid = data[i]["paragraphs"][x]["qas"][y]["id"]
                
                if(data[i]["paragraphs"][x]["qas"][y]["answer"] != ""):
                    pos = int(data[i]["paragraphs"][x]["qas"][y]["answer"]["answer_start"])
                    length = len(data[i]["paragraphs"][x]["qas"][y]["answer"]["text"].split(" "))
                    if LABEL_POS in f:
                        f[LABEL_POS].append(pos)
                    else:
                        f[LABEL_POS] = [pos]
                        
                    if LABEL_LEN in f:
                        f[LABEL_LEN].append(length)
                    else:
                        f[LABEL_LEN] = [length]
                    
                ids.append(mid)
                
                if FEATURES[0] in f:
                    f[FEATURES[0]].append(c)
                else:
                    f[FEATURES[0]] = [c]
                
                if FEATURES[1] in f:
                    f[FEATURES[1]].append(qn)
                else:
                    f[FEATURES[1]] = [qn]
    return f, ids;

def input_pos(data_set):
    feature_cols = {k: tf.string_to_hash_bucket_fast(data_set[k], 1) for k in FEATURES}
    labels = None
    if LABEL_POS in data_set:
        labels = tf.constant(data_set[LABEL_POS], dtype=tf.int32)
    return feature_cols, labels
    
def input_len(data_set):
    feature_cols = {k: tf.string_to_hash_bucket_fast(data_set[k], 1) for k in FEATURES}
    labels = None
    if LABEL_LEN in data_set:
        labels = tf.constant(data_set[LABEL_LEN], dtype=tf.int32)
    return feature_cols, labels
    
def getModelPos():
    feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                            hidden_units=[100, 200, 100],
                                            optimizer=tf.train.AdadeltaOptimizer(
                                                    learning_rate=0.01,
                                                    rho=0.001, 
                                                    epsilon=0.1, 
                                                    use_locking=False, 
                                                    name='AdaDelta'
                                            ),
                                            model_dir=model_root + model_pos);
    return regressor

def getModelLen():
    feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                            hidden_units=[100, 200, 100],
                                            optimizer=tf.train.AdadeltaOptimizer(
                                                    learning_rate=0.01,
                                                    rho=0.001, 
                                                    epsilon=0.1, 
                                                    use_locking=False, 
                                                    name='AdaDelta'
                                            ),
                                            model_dir=model_root + model_len);
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

data_set, ids = getData('train.json');

m_pos = getModelPos();
m_len = getModelLen();

m_pos.fit(input_fn=lambda:input_pos(data_set), steps=1000)
m_len.fit(input_fn=lambda:input_len(data_set), steps=1000)

pos_preds = m_pos.predict(input_fn=lambda:input_pos(data_set), as_iterable=False);
len_preds = m_len.predict(input_fn=lambda:input_len(data_set), as_iterable=False);

data = "Id,Answer\n"
for i in range(len(ids)):
    mid = ids[i]
    length = int(len_preds[i]);
    if length == 0:
        length = 2
    start = int(pos_preds[i]);
    m_context = data_set[FEATURES[0]][i][start:];
    arr = m_context.split(" ");
    
    stri = ""
    for x in range(1, length):
        stri += arr[x] + " ";
        
    data += mid + "," + normalize_answer(stri) + "\n"

data = data.encode('utf-8')
with open('answers.csv','wb') as file:
    file.write(data)

data_set, ids = getData('test.json');

pos_preds = m_pos.predict(input_fn=lambda:input_pos(data_set), as_iterable=False);
len_preds = m_len.predict(input_fn=lambda:input_len(data_set), as_iterable=False);

data = "Id,Answer\n"
for i in range(len(ids)):
    mid = ids[i]
    length = int(len_preds[i]) + 1;
    if length == 1:
        length += 1
    start = int(pos_preds[i]);
    m_context = data_set[FEATURES[0]][i][start:];
    arr = m_context.split(" ");
    
    stri = ""
    for x in range(1, length):
        stri += arr[x] + " ";
        
    data += mid + "," + normalize_answer(stri) + "\n"

data = data.encode('utf-8')
with open('submission.csv','wb') as file:
    file.write(data)
