import matplotlib
import numpy
import os
import pdb
import scipy

from matplotlib import pyplot
from scipy import stats

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

norms = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            header = line.copy()
            counter += 1
            continue
        assert len(line) == len(header)
        #marker = False
        for k in relevant_keys:
            if float(line[header.index(k)]) > 10:
                line[header.index(k)] = '.{}'.format(line[header.index(k)])
            #try:
            assert float(line[header.index(k)]) < 10
            #except AssertionError:
            #    #logging.info(line[0])
            #    marker = True
        #if marker:
        #    continue
        if len(line[0].split()) == 1:
            norms[line[0].lower()] = list()
            for k in relevant_keys:
                val = float(line[header.index(k)])
                norms[line[0].lower()].append(val)

damages = [
           'full_random',
           'noise_injection',
           'auditory_relu-raw-thresholded9095_random',
           ]

for damage in damages:
    print('\n')
    for dataset in ['1', '2']:
        ### open results
        words = list()
        undamaged = list()
        #with open(os.path.join('brain_results', 'fernandino2_undamaged_w2v_en.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino1_undamaged_w2v_en.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino2_undamaged_w2v_en.results')) as i:
        with open(os.path.join('brain_results', 'rsa_fernandino{}_undamaged_w2v_en.results'.format(dataset))) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                words.append(line[0])
                undamaged.append(line[1:])
        undamaged = numpy.array(undamaged, dtype=numpy.float64)

        ### open results
        dam_words = list()
        damaged = list()
        #with open(os.path.join('brain_results', 'rsa_fernandino1_w2v_en_auditory_relu-raw95_random.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino1_w2v_en_auditory_relu-raw-thresholded9095_random.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino2_w2v_en_auditory_relu-raw-thresholded9095_random.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino2_w2v_en_auditory_relu-raw-thresholded8595_random.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino2_w2v_en_noise_injection.results')) as i:
        #with open(os.path.join('brain_results', 'rsa_fernandino1_w2v_en_noise_injection.results')) as i:
        with open(os.path.join('brain_results', 'rsa_fernandino{}_w2v_en_{}.results'.format(dataset, damage))) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                dam_words.append(line[0])
                damaged.append(line[1:])
        damaged = numpy.array(damaged, dtype=numpy.float64)
        assert dam_words == words
        #high_auditory = [w_i for w_i, w in enumerate(words) if norms[w][relevant_keys.index('Auditory.mean')]>4.]
        #high_auditory = [w_i for w_i, w in enumerate(words) if norms[w][relevant_keys.index('Visual.mean')]>=3.]
        #high_auditory = [w_i for w_i, w in enumerate(words) if norms[w][relevant_keys.index('Mouth.mean')]>=4.]
        high_auditory = [w_i for w_i, w in enumerate(words)]
        diff = scipy.stats.wilcoxon(
                                    numpy.average(damaged[high_auditory, :], axis=0), 
                                    numpy.average(undamaged[high_auditory, :], axis=0), 
                                    alternative='less'
                                    #alternative='greater'
                                    )
        #diff = scipy.stats.wilcoxon(damaged[high_auditory, :].flatten(), undamaged[high_auditory, :].flatten(), alternative='less')
        print(damage)
        print(dataset)
        print(diff)
import pdb; pdb.set_trace()
