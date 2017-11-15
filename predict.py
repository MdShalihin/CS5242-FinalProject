from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin
from config import Config


from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data
from data_utils import *

import logging
logging.basicConfig(level=logging.INFO)

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset)), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset[articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data



def prepare_dev2(config):
    dev = squad_dataset(config.question_dev, config.context_dev, config.answer_dev)



def prepare_dev(prefix, dev_filename, vocab):
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)
    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    def normalize(dat):
        return map(lambda tok: map(int, tok.split()), dat)

    context_data = normalize(context_data)
    question_data = normalize(question_data)

    return context_data, question_data, question_uuid_data


def generate_answers(sess, model, dataset, uuid_data, rev_vocab):
    answers = {}

    q,c,a = dataset
    num_points = len(a)
    sample_size = 1000


    answers_canonical = []
    num_iters = int((num_points+sample_size-1)/sample_size)

    for i in xrange(num_iters):
        curr_slice_st = i*sample_size
        curr_slice_en = min((i+1)*sample_size, num_points)

        slice_sz = curr_slice_en - curr_slice_st 

        q_curr = q[curr_slice_st : curr_slice_en]
        c_curr = c[curr_slice_st : curr_slice_en]
        a_curr = a[curr_slice_st : curr_slice_en]

        s, e = model.answer(sess, [q_curr, c_curr, a_curr])

        for j in xrange(slice_sz):
            st_idx = s[j]
            en_idx = e[j]
            curr_context = c[curr_slice_st+j]
            curr_uuid = uuid_data[curr_slice_st+j]

            curr_ans = ""
            for idx in xrange(st_idx, en_idx+1):
                curr_tok = curr_context[idx]
                curr_ans += " %s" %(rev_vocab[curr_tok])

            answers[curr_uuid] = curr_ans
            answers_canonical.append((s,e))

    return answers, answers_canonical


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

def run_func():
    config = Config()
    vocab, rev_vocab = initialize_vocab(config.vocab_path)

    dev_path = "download/squad/test.json"
    dev_dirname = os.path.dirname(os.path.abspath(dev_path))
    dev_filename = os.path.basename(dev_path)
    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)


    
    ques_len = len(question_data)
    answers = [[0, 0] for _ in xrange(ques_len)]

    dataset = [question_data, context_data, answers]

    embed_path = config.embed_path

    embeddings = get_trimmed_glove_vectors(embed_path)


    encoder = Encoder(config.hidden_state_size)
    decoder = Decoder(config.hidden_state_size)

    qa = QASystem(encoder, decoder, embeddings, config)
    
    data = "Id,Answer\n"
    
    with tf.Session() as sess:
        qa.initialize_model(sess, config.train_dir)
        answers, _ = generate_answers(sess, qa, dataset, question_uuid_data, rev_vocab)
        for a in answers:
            ans = answers[a]
            data += a + "," + normalize_answer(ans).replace(" s ", "s ") + "\n"

    with open('submission.csv','wb') as file:
        file.write(data)


if __name__ == "__main__":
    run_func()
