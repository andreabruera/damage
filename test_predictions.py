import argparse
import fasttext
import gensim
import logging
import numpy
import os
import pickle
import random
import scipy
import sklearn

from gensim.models import Word2Vec
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--language', choices=['it', 'en', 'de'], required=True)
parser.add_argument('--model', choices=['fasttext', 'w2v'], required=True)
args = parser.parse_args()

### reading frequencies
ratings = dict()
for corpus in ['opensubs', 'wac']:
    freqs = pickle.load(open(os.path.join('..', 'psychorpus', 'pickles', args.language, '{}_{}_word_freqs.pkl'.format(args.language, corpus)), 'rb'))
    for original_k, v in freqs.items():
        k = original_k.lower()
        ### frequencies are not lowercased
        if k not in ratings.keys():
            ratings[k] = v
        else:
            ratings[k] += v

overall_keys = list()

relevant_keys = ['Conc.M']
overall_keys.extend(relevant_keys)
file_path = os.path.join(
                         'data',
                         'brysbaert_conc.tsv',
                         )
assert os.path.exists(file_path)
norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.replace(',', '.').strip().split('\t')
        if counter == 0:
            header = line.copy()
            counter += 1
            continue
        assert len(line) == len(header)
        word = line[0].lower()
        #if word in ratings.keys():
        if len(word.split()) == 1:
            for k in relevant_keys:
            #if ratings[word] > 1000:
                norms[word] = [line[header.index(k)]]


logging.info(len(norms))

file_path = os.path.join(
                         'data',
                         'Lancaster_sensorimotor_norms_for_39707_words.tsv',
                         )
assert os.path.exists(file_path)
relevant_keys = [
                 'Auditory.mean',
                 'Gustatory.mean',
                 'Haptic.mean',
                 'Olfactory.mean',
                 'Visual.mean',
                 'Foot_leg.mean',
                 'Hand_arm.mean', 
                 'Head.mean', 
                 'Mouth.mean', 
                 'Torso.mean'
                 ]
overall_keys.extend(relevant_keys)

#norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            #logging.info(line)
            header = line.copy()
            counter += 1
            continue
        assert len(line) == len(header)
        word = line[0].lower()
        #if word in ratings.keys():
        #    if len(word.split()) == 1:
        #        if ratings[word] > 1000:
        #            norms[word] = line[1:]
        ### checking because some lines contain errors
        marker = False
        for k in relevant_keys:
            try:
                assert float(line[header.index(k)]) <= 5
            except AssertionError:
                marker = True
        if marker:
            continue

        if word in norms.keys():
            for k in relevant_keys:
            #if ratings[word] > 1000:
                norms[word].append(line[header.index(k)])
            #logging.info(overall_keys)
            #logging.info(norms[word])

'''
file_path = os.path.join(
                         'data',
                         #'lynott_perceptual.tsv',
                         'binder_ratings.tsv',
                         )
assert os.path.exists(file_path)
relevant_keys = [
        'Audition', 'Loud', 'Low', 'High','Sound', 'Music', 'Speech'
                 ]
overall_keys.extend(relevant_keys)

#norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.replace(',', '.').strip().split('\t')
        if counter == 0:
            #header = line.copy()[1:]
            header = line.copy()
            counter += 1
            continue
        #assert len(line[1:]) == len(header)
        word = line[1].lower()
        #if word in ratings.keys():
            #if len(word.split()) == 1:
                #if ratings[word] > 1000:
                #norms[word] = line[1:]
        if word in norms.keys():
            for k in relevant_keys:
            #if ratings[word] > 1000:
                norms[word].append(line[header.index(k)])

logging.info(len(norms))

### lynott
file_path = os.path.join(
                         'data',
                         'lynott_perceptual.tsv',
                         #'binder_ratings.tsv',
                         )
assert os.path.exists(file_path)

relevant_keys =  [
                 'Auditory_mean',
                 'Gustatory_mean',
                 'Haptic_mean',
                 'Olfactory_mean',
                 'Visual_mean',
                 ]
overall_keys.extend(relevant_keys)
#norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.replace(',', '.').strip().split('\t')
        if counter == 0:
            #header = line.copy()[1:]
            header = line.copy()
            counter += 1
            continue
        #assert len(line[1:]) == len(header)
        word = line[1].lower()
        #if word in ratings.keys():
            #if len(word.split()) == 1:
                #if ratings[word] > 1000:
                #norms[word] = line[1:]
        if word in norms.keys():
            for k in relevant_keys:
            #if ratings[word] > 1000:
                norms[word].append(line[header.index(k)])
#logging.info(len(norms))

'''

norms = {k : v for k,v in norms.items() if len(v)==len(overall_keys)}
logging.info(len(norms))
### loading fasttext model

if args.model == 'fasttext':
    logging.info('now loading fasttext')
    model = fasttext.load_model(os.path.join(
                                    '/',
                                    'import',
                                    'cogsci',
                                    'andrea',
                                    'dataset',
                                    'word_vectors',
                                    'en',
                                    'cc.en.300.bin',
                                    )
                                    )
    model_vocabulary = model.words
if args.model == 'w2v':
    logging.info('now loading w2v')
    model = Word2Vec.load(os.path.join(
                                    #'/',
                                    #'import',
                                    #'cogsci',
                                    #'andrea',
                                    #'dataset',
                                    #'word_vectors',
                                    #'en',
                                    'models',
                                    'word2vec_en_damaged_auditory_relu_param-mandera2017_min-count-50.model'
                                    #'word2vec_en_damaged_auditory_raw_param-mandera2017_min-count-50.model'
                                    #'word2vec_en_damaged_auditory_relu-exponential_param-mandera2017_min-count-50.model'
                                    #'word2vec_en_damaged_auditory_logarithmic_param-mandera2017_min-count-50.model'
                                    #'word2vec_en_damaged_auditory_sigmoid_param-mandera2017_min-count-50.model'
                                    #'word2vec_en_damaged_auditory_relu-logarithmic_param-mandera2017_min-count-50.model'
                                    )
                                    ).wv
    model_vocabulary = [w for w in model.vocab]

logging.info('now preparing the training data')
all_words = [w for w in norms.keys() if w in model_vocabulary]

all_vectors = [model[w] for w in all_words]
for vec in all_vectors:
    assert vec.shape == (300, )

all_targets = [numpy.array(norms[w], dtype=numpy.float32) for w in all_words]
for vec in all_targets:
    #assert vec.shape == (7, )
    assert vec.shape == (len(overall_keys), )
    #assert vec.shape == (5, )
    #assert vec.shape == (1, )

### 100 times random combinations
twenty = int(len(all_words)*0.2)

results = {k : list() for k in overall_keys}
logging.info('now training/testing...')
for i in tqdm(range(20)):
    test_items = random.sample(all_words, k=twenty)
    training_input = [all_vectors[v_i] for v_i, v in enumerate(all_words) if v not in test_items]
    training_target = [all_targets[v_i] for v_i, v in enumerate(all_words) if v not in test_items]
    test_input = [all_vectors[all_words.index(v)] for v in test_items]
    test_target = [all_targets[all_words.index(v)] for v in test_items]
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
    #ridge = Ridge()
    ridge.fit(training_input, training_target)
    predictions = ridge.predict(test_input)
    for key_i, key in enumerate(overall_keys):
        ### computing correlations
        real = [target[key_i] for target in test_target]
        preds = [pred[key_i] for pred in predictions]
        corr = scipy.stats.pearsonr(real, preds)[0]
        results[key].append(corr)
logging.info('model : {}'.format(args.model))
for k, v in results.items():
    logging.info('correlation for {}: {}'.format(k,numpy.nanmean(v)))
