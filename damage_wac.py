import argparse
import fasttext
import numpy
import os
import pickle
import sklearn

from sklearn.linear_model import RidgeCV
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--corpora_path', 
                    required=True,
                    help='path to the folder containing '
                    'the files for all the languages/corpora'
                    )
parser.add_argument('--language', choices=['it', 'en', 'de'], required=True)
args = parser.parse_args()

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
        if len(line[0].split()) == 1:
            norms[line[0].lower()] = line[1:]

relevant_keys = [
                 'Auditory.mean',
                 'Gustatory.mean',
                 'Haptic.mean',
                 'Olfactory.mean',
                 'Visual.mean',
                 ]

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

print('now reducing the vocabulary')
reduced_vocab = [k for k, v in ratings.items() if v > 50]
print(len(reduced_vocab))

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

print('now preparing the training data')
training_words = [w for w in norms.keys() if w in ft.words]
training_input = [ft[w] for w in training_words]
for vec in training_input:
    assert vec.shape == (300, )
training_target = [numpy.array([norms[w][header.index(idx)] for idx in relevant_keys], dtype=numpy.float32) for w in training_words]
for vec in training_target:
    assert vec.shape == (5, )

print('now preparing the test data')
to_be_predicted_words = [w for w in reduced_vocab if w not in training_words and w in ft.words]
to_be_predicted_input = [ft[w] for w in to_be_predicted_words]
for vec in to_be_predicted_input:
    assert vec.shape == (300, )

print('now training...')
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
ridge.fit(training_input, training_target)
print('...finally predicting and writing to file!')
predictions = ridge.predict(to_be_predicted_input)

out_folder = os.path.join('predictions', 'sensory')
os.makedirs(out_folder, exist_ok=True)

with open(os.path.join(out_folder, 'sensory_training_fasttext.tsv'), 'w') as o:
    o.write('word\t')
    for k in relevant_keys:
        o.write('word\t')
    o.write('\n')
    for k, v in zip(training_words, training_target):
        o.write('{}\t'.format(k))
        for pred in v:
            o.write('{}\t'.format(pred))

with open(os.path.join(out_folder, 'sensory_predicted_fasttext.tsv'), 'w') as o:
    o.write('word\t')
    for k in relevant_keys:
        o.write('word\t')
    o.write('\n')
    for k, v in zip(to_be_predicted_words, predictions):
        o.write('{}\t'.format(k))
        for pred in v:
            o.write('{}\t'.format(pred))

import pdb; pdb.set_trace()

### loading wac
if args.language == 'en':
    identifier = 'PukWaC'
elif args.language == 'it':
    identifier = 'itwac'
elif args.language == 'de':
    identifier = 'sdewac-v3-tagged'
wac_folder = os.path.join(args.corpora_path, args.language, '{}_smaller_files'.format(identifier))
assert os.path.exists(wac_folder)
### loading opensubs
opensubs_folder = os.path.join(args.corpora_path, args.language, 'opensubs_ready')
assert os.path.exists(opensubs_folder)
