import os

import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

import utils


PATH = r"results\mnist\iter0\recordings"
W_START = 150e-3
W_END = 200e-3
TST_SPLIT = 0.15

def load_sample(path):
	d = utils.load_dict(path)
	spike_times, channels = d['processed']['spike_times'], d['processed']['channels']

	stim_time = d['stim_time']
	fps = d['fps']
	sample = np.zeros(4096, dtype=np.float32)
	for i, st in enumerate(spike_times):
		if st >= stim_time + W_START*fps and st <= stim_time + W_END*fps: sample[channels[i]] += 1
	return sample

if __name__ == '__main__':
	names, X, Y = [], [], []

	print("Loading data...")
	classes = {c: cls for c, cls in enumerate(sorted(os.listdir(PATH)))}
	for c, cls in classes.items():
		files = [f for f in os.listdir(os.path.join(PATH, cls)) if f.endswith('.pt')]
		for f in files:
			sample = load_sample(os.path.join(PATH, cls, f))
			names.append(f)
			X.append(sample)
			Y.append(c)

	names, X, Y = shuffle(names, X, Y, random_state=0)
	num_tst = int(TST_SPLIT * len(X))
	X_trn, Y_trn = X[:-num_tst], Y[:-num_tst]
	X_tst, Y_tst = X[-num_tst:], Y[-num_tst:]
	print("Train size: {}, test size: {}".format(len(X_trn), len(X_tst)))

	selected = 0
	print("Plotting sample {} of class {}".format(names[selected], Y[selected]))
	plt.imshow((X[selected]/max(X[selected])).reshape(64, 64))
	plt.show()

	clf = SGDClassifier(class_weight='balanced')
	clf.fit(X_trn, Y_trn)
	score = clf.score(X_tst, Y_tst)

	print(score)





