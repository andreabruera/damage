import argparse
import numpy
import os
import pickle

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
            norms[line[0]] = line[1:]

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
reduced_vocab = [k for k, v in ratings.items() if v > 50]
print(len(reduced_vocab))

with tqdm() as counter:
    for w in norms.keys():
        w_lower = w.lower()
        if w_lower not in reduced_vocab:
            #print(w_lower)
            counter.update(1)
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
