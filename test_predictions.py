import argparse
import fasttext
import numpy
import os
import pickle
import random
import scipy
import sklearn

from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--language', choices=['it', 'en', 'de'], required=True)
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

'''
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
            header = line.copy()[1:]
            counter += 1
            continue
        assert len(line[1:]) == len(header)
        word = line[0].lower()
        if word in ratings.keys():
            if len(word.split()) == 1:
                #if ratings[word] > 1000:
                norms[word] = line[1:]
relevant_keys = ['Conc.M']
file_path = os.path.join(
                         'data',
                         'Lancaster_sensorimotor_norms_for_39707_words.tsv',
                         )
assert os.path.exists(file_path)

norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            header = line.copy()[1:]
            counter += 1
            continue
        assert len(line[1:]) == len(header)
        word = line[0].lower()
        if word in ratings.keys():
            if len(word.split()) == 1:
                if ratings[word] > 1000:
                    norms[word] = line[1:]

relevant_keys = [
                 'Auditory.mean',
                 'Gustatory.mean',
                 'Haptic.mean',
                 'Olfactory.mean',
                 'Visual.mean',
                 ]
'''
file_path = os.path.join(
                         'data',
                         #'lynott_perceptual.tsv',
                         'binder_ratings.tsv',
                         )
assert os.path.exists(file_path)

norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.replace(',', '.').strip().split('\t')
        if counter == 0:
            header = line.copy()[1:]
            counter += 1
            continue
        assert len(line[1:]) == len(header)
        word = line[0].lower()
        if word in ratings.keys():
            if len(word.split()) == 1:
                #if ratings[word] > 1000:
                norms[word] = line[1:]

relevant_keys = [
        'Audition', 'Loud', 'Low', 'High','Sound', 'Music', 'Speech'
                 ]

### loading fasttext model

print('now loading fasttext')
ft = fasttext.load_model(os.path.join(
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

print('now preparing the data')
all_words = [w for w in norms.keys() if w in ft.words]

all_vectors = [ft[w] for w in all_words]
for vec in all_vectors:
    assert vec.shape == (300, )

all_targets = [numpy.array([norms[w][header.index(idx)] for idx in relevant_keys], dtype=numpy.float32) for w in all_words]
for vec in all_targets:
    assert vec.shape == (7, )
    #assert vec.shape == (5, )
    #assert vec.shape == (1, )

### 100 times random combinations
twenty = int(len(all_words)*0.2)

results = {k : list() for k in relevant_keys}
print('now training/testing...')
for key_i, key in enumerate(relevant_keys):
    for i in tqdm(range(10)):
        test_items = random.sample(all_words, k=twenty)
        training_input = [all_vectors[v_i] for v_i, v in enumerate(all_words) if v not in test_items]
        training_target = numpy.array([[all_targets[v_i][key_i]] for v_i, v in enumerate(all_words) if v not in test_items], dtype=numpy.float32)
        test_input = [all_vectors[all_words.index(v)] for v in test_items]
        test_target = numpy.array([[all_targets[all_words.index(v)][key_i]] for v in test_items], dtype=numpy.float32)
        ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
        #ridge = Ridge()
        ridge.fit(training_input, training_target)
        predictions = ridge.predict(test_input)
        ### computing correlations
        #real = [target[key_i] for target in test_target]
        #preds = [pred[key_i] for pred in predictions]
        #corr = scipy.stats.pearsonr(real, preds)[0]
        corr = scipy.stats.pearsonr(test_target.reshape(-1), predictions)[0]
        results[key].append(corr)
    print('correlation for {}: {}'.format(key,numpy.nanmean(results[key])))
#for k, v in results.items():
#    print('correlation for {}: {}'.format(k,numpy.nanmean(v)))
