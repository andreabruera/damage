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
           #'full_random',
           #'noise_injection',
           #'auditory_relu-raw-thresholded9095_random',
           #'auditory_relu-raw90_random',
           #'auditory_relu-raw75_random',
           'auditory_relu-raw95_random',
           #'auditory_relu-exponential75_random',
           #'auditory_relu-exponential90_random',
           #'auditory_relu-exponential95_random',
           ]

plot_folder = os.path.join('brain_plots', 'rsa')
os.makedirs(plot_folder, exist_ok=True)

for damage in damages:
    print('\n')
    for dataset in ['1', '2']:
        ### open results
        dam_results = list()
        with open(os.path.join('brain_results', 'bins_rsa_fernandino{}_w2v_en_{}.results'.format(dataset, damage))) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                dam_results.append(numpy.array(line[1:], dtype=numpy.float32))
        ### open results
        undam_results = list()
        with open(os.path.join('brain_results', 'bins_rsa_fernandino{}_undamaged_w2v_en.results'.format(dataset))) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                undam_results.append(numpy.array(line[1:], dtype=numpy.float32))
        diff = scipy.stats.ttest_rel(
                                    numpy.average(dam_results, axis=0), 
                                    numpy.average(undam_results, axis=0), 
                                    alternative='less'
                                    #alternative='greater'
                                    )
        print(damage)
        print(dataset)
        print(diff)
        ### plotting
        fig, ax = pyplot.subplots(constrained_layout=True)
        mod_xs = list(range(5))
        mod_dam_ys = numpy.average(dam_results, axis=1)
        mod_undam_ys = numpy.average(undam_results, axis=1)
        ax.plot(mod_xs, mod_dam_ys, color='orange', label='damaged auditory')
        ax.plot(mod_xs, mod_undam_ys, color='grey', ls='-.', label='undamaged')
        ax.set_xlim(left=0., right=5.)
        ax.legend()
        file_out = os.path.join(plot_folder, 'bins_brain_rsa_fernandino{}_{}.jpg'.format(dataset, damage))
        print(file_out)
        pyplot.savefig(file_out)
        pyplot.clf()
        pyplot.close()
        

import pdb; pdb.set_trace()
