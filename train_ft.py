import fasttext
import os

folder = os.path.join('/', 'import', 'cogsci', 'andrea','dataset', 'corpora', 'de', 'ready_for_fasttext')
out_folder = folder.replace('corpora', 'word_vectors')
for f in sorted(os.listdir(folder)[6:], reverse=True):
    file_path = os.path.join(folder, f)
    out_folder = folder.replace('corpora', 'word_vectors')
    out_file = os.path.join(out_folder, f.replace('txt', 'bin'))
    if os.path.exists(out_file):
        continue
    os.makedirs(out_folder, exist_ok=True)
    model = fasttext.train_unsupervised(file_path, model='skipgram', dim=300, minCount=50, thread=int((os.cpu_count()/3)*2))
    model.save_model(out_file)
