import os

import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

import utils


PATH = r"results\mnist\iter0\recordings0"
W_START = 10e-3
W_END = 50e-3
TST_SPLIT = 0.15

def load_sample(path):
	d = utils.load_dict(path)
	input = d['input']
	spike_times, channels = d['processed']['spike_times'], d['processed']['channels']

	#stim_time = d['stim_time']
	stim_time = min([st for i, st in enumerate(spike_times) if channels[i] == 1])
	fps = d['fps']
	sample = np.zeros(4096, dtype=np.float32)
	for i, st in enumerate(spike_times):
		if st >= stim_time + W_START*fps and st <= stim_time + W_END*fps: sample[channels[i]] += 1
	return input, sample

if __name__ == '__main__':
	names, X_in, X, Y = [], [], [], []

	print("Loading data...")
	classes = {c: cls for c, cls in enumerate(sorted(os.listdir(PATH)))}
	for c, cls in classes.items():
		files = [f for f in os.listdir(os.path.join(PATH, cls)) if f.endswith('.pt')]
		for f in files:
			input, sample = load_sample(os.path.join(PATH, cls, f))
			names.append(f)
			X_in.append(input.reshape(-1).tolist())
			X.append(sample)
			Y.append(c)

	names, X_in, X, Y = shuffle(names, X_in, X, Y, random_state=0)
	num_tst = int(TST_SPLIT * len(X))
	X_in_trn, X_trn, Y_trn = X_in[:-num_tst], X[:-num_tst], Y[:-num_tst]
	X_in_tst, X_tst, Y_tst = X_in[-num_tst:], X[-num_tst:], Y[-num_tst:]
	print("Train size: {}, test size: {}".format(len(X_trn), len(X_tst)))

	selected = 0
	print("Plotting sample {} of class {}".format(names[selected], Y[selected]))
	plt.imshow((X[selected]/max(X[selected])).reshape(64, 64))
	plt.show()

	print("Evaluating classifier on input...")
	clf = SGDClassifier(class_weight='balanced')
	clf.fit(X_in_trn, Y_trn)
	score = clf.score(X_in_tst, Y_tst)
	print("Accuracy: {}".format(score))

	print("Evaluating classifier on biological features...")
	clf = SGDClassifier(class_weight='balanced')
	clf.fit(X_trn, Y_trn)
	score = clf.score(X_tst, Y_tst)
	print("Accuracy: {}".format(score))

