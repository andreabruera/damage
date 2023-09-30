import argparse
import gensim
import logging
import math
import multiprocessing
import numpy
import os
import pickle
import random
import sklearn
import stop_words
import string

from gensim.models import Word2Vec
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from utils import prepare_input_output_folders

class DamagedCorpora:
    def __init__(self, file_names):
        self.file_names = file_names
    def __iter__(self):
        for file_name in self.file_names:
            with open(file_name) as i:
                for l in i:
                    line = l.strip().split()
                    if len(line) > 3:
                        yield line

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--corpora_path', 
                    required=True,
                    help='path to the folder containing '
                    'the files for all the languages/corpora'
                    )
parser.add_argument('--language', choices=['it', 'en', 'de'], required=True)
parser.add_argument('--semantic_modality', choices=['auditory', 'action'], required=True)
parser.add_argument('--function', choices=['sigmoid', 'raw', 'exponential', 'relu', 'relu-exponential', 'logarithmic', 'relu-logarithmic', 'relu-sigmoid'], required=True)
parser.add_argument('--relu_base', choices=['50', '75', '90'], default='90')
parser.add_argument('--debugging', action='store_true')
args = parser.parse_args()

logging.info('now loading the folders!')
file_names = prepare_input_output_folders(args)

model = Word2Vec(
                 sentences=DamagedCorpora(file_names), 
                 size=300, 
                 window=6, 
                 workers=int(os.cpu_count()),
                 min_count=50,
                 negative=10,
                 sg=0,
                 sample=1e-5,
                 )
os.makedirs('models', exist_ok=True)
model.save(os.path.join('models', "word2vec_{}_damaged_{}_{}_param-mandera2017_min-count-50.model".format(args.language, args.semantic_modality, args.function)))
