import os
import argparse
import math
import random
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset

from data import CustomMNIST as MNIST, CustomCIFAR10 as CIFAR10, CustomCIFAR100 as CIFAR100, get_transform
import utils
from params import *


class Experiment:
	def __init__(self, exp_name, dataset, mode, seed, dataseed):
		print("Initializing experiment...")

		self.exp_name = exp_name
		self.dataset = dataset
		self.mode = mode
		self.seed = seed
		self.dataseed = dataseed
		self.data_folder = 'datasets'
		self.exp_folder = os.path.join('json_datasets', self.exp_name, 'iter{}'.format(self.seed))
		self.json_path = os.path.join(self.exp_folder, 'json_files')
		self.data_perm_save_path = os.path.join(self.exp_folder, 'dataperm.json')

		# Preparing data
		self.data_len = None
		utils.set_rng_seed(self.dataseed)
		self.trn_set, self.tst_set = self.get_trn_tst_sets()

		# Initializing experiment state
		utils.set_rng_seed(self.seed)

	def get_trn_tst_sets(self):
		trn_dataset, tst_dataset = None, None
		if self.dataset == 'mnist':
			trn_dataset = MNIST(root=os.path.join(self.data_folder, 'mnist'), train=True, transform=get_transform(detect_edges=DETECT_EDGES), download=True)
			tst_dataset = MNIST(root=os.path.join(self.data_folder, 'mnist'), train=False, transform=get_transform(detect_edges=DETECT_EDGES), download=True)
		elif self.dataset == 'cifar10':
			trn_dataset = CIFAR10(root=os.path.join(self.data_folder, 'mnist'), train=True, transform=get_transform(detect_edges=DETECT_EDGES), download=True)
			tst_dataset = CIFAR10(root=os.path.join(self.data_folder, 'mnist'), train=False, transform=get_transform(detect_edges=DETECT_EDGES), download=True)
		elif self.dataset == 'cifar100':
			trn_dataset = CIFAR100(root=os.path.join(self.data_folder, 'mnist'), train=True, transform=get_transform(detect_edges=DETECT_EDGES), download=True)
			tst_dataset = CIFAR100(root=os.path.join(self.data_folder, 'mnist'), train=False, transform=get_transform(detect_edges=DETECT_EDGES), download=True)
		if trn_dataset is None or tst_dataset is None:
			raise ValueError("Unsupported dataset {}. Only mnist, cifar10, cifar100 are supported.".format(self.dataset))
		self.data_len = len(trn_dataset) + len(tst_dataset)
		#return self._get_trn_tst_sets(trn_dataset, tst_dataset)
		return self._get_trn_tst_sets_slice(trn_dataset, tst_dataset)

	def _get_trn_tst_sets(self, trn_dataset, tst_dataset):
		return DataLoader(trn_dataset, batch_size=1, shuffle=True, num_workers=1), DataLoader(tst_dataset, batch_size=1, shuffle=False, num_workers=1)

	def _get_trn_tst_sets_slice(self, trn_dataset, tst_dataset):
		n_train, n_test = len(trn_dataset), len(tst_dataset)
		cls_indices = {cls: torch.nonzero(trn_dataset.targets == cls).reshape(-1).tolist()[:TRN_SAMPLES_PER_CLASS] for cls in range(len(trn_dataset.classes))}
		trn_indices = [idx for cls, idxs in cls_indices.items() for idx in idxs]
		chosen_trn_indices = [idx for cls, idxs in cls_indices.items() if cls in [0, 1] for idx in idxs]
		random.shuffle(chosen_trn_indices)
		trn_perm, tst_perm = chosen_trn_indices, [i for i in range(len(trn_dataset)) if i not in trn_indices]
		trn_dataset_shuffled, trn_dataset = Subset(trn_dataset, chosen_trn_indices), Subset(trn_dataset, tst_perm)
		tst_dataset = ConcatDataset([trn_dataset, tst_dataset])
		tst_indices = list(range(len(tst_dataset)))
		random.shuffle(tst_indices)
		tst_perm = [tst_perm[i] if i < len(tst_perm) else n_train + i for i in tst_indices]
		tst_indices = tst_indices[TST_SAMPLES[0]:TST_SAMPLES[1]]
		tst_dataset_shuffled = Subset(tst_dataset, tst_indices)
		utils.save_data_permutation(self.data_perm_save_path, trn_perm, tst_perm)
		return DataLoader(trn_dataset_shuffled, batch_size=1, shuffle=True, num_workers=1), DataLoader(tst_dataset_shuffled, batch_size=1, shuffle=False, num_workers=1)

	def save_json(self, index, signal, label):
		with open('template_json.json') as f:
			json_data = json.load(f)

		# Convert encoded input to spike signal
		spikes = torch.bernoulli(signal*MAX_FIRING_LIKELIHOOD)
		spikes = torch.repeat_interleave(spikes, repeats=2, dim=-1)
		spikes[:, 1::2] *= -1
		idx_pos = [l.item()+1 for l in torch.nonzero(spikes.reshape(-1) == 1)]
		idx_neg = [l.item()+1 for l in torch.nonzero(spikes.reshape(-1) == -1)]

		json_data["Groups"][0]["PixelIndexes"] = idx_pos
		json_data["Groups"][1]["PixelIndexes"] = idx_neg

		path = os.path.join(self.json_path, '{}'.format(label), '{}.json'.format(index))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, 'w+') as f:
			json.dump(json_data, f)

	def process_sample(self, index, input, label):
		# Encode input
		signal, masked_signal = utils.gen_signal(input, label)
		if self.mode == 'train':
			self.save_json(index, signal, label)
		else:
			self.save_json(index, masked_signal, label)

	def process_epoch(self, dataset):
		# Iterate through dataset, send inputs to device, record and process outputs
		for indexes, inputs, labels in tqdm(dataset):
			for index, input, label in zip(indexes, inputs, labels):
				self.process_sample(str(index.item()).zfill(int(math.log10(self.data_len))+1), input, label)

	def train(self):
		# Train
		print("\nTRAIN | Experiment {} | Iter {}".format(self.exp_name, self.seed))
		print("Training...")
		self.process_epoch(self.trn_set)

	def eval(self):
		# Test
		print("EVAL | Experiment {} | Iter {}".format(self.exp_name, self.seed))
		print("Testing...")
		self.process_epoch(self.tst_set)


def run_experiment(name, dataset, mode, seeds, dataseeds):
	for iter, seed in enumerate(seeds):
		print("\n####    Iter {}    ####".format(seed))
		dataseed = dataseeds[iter % len(dataseeds)]

		if 'train' in mode: Experiment(name, dataset, 'train', seed, dataseed).train()
		if 'test' in mode: Experiment(name, dataset, 'test', seed, dataseed).eval()
		print("\nIter {} completed!".format(seed))

	print("\n\nFinished!")


if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help="Name assigned to the experiment")
	parser.add_argument('--dataset', default='mnist', type=str, help="Dataset name to be used for the experiment")
	parser.add_argument('--mode', default='test', choices=['train', 'test', 'traintest'], help="Whether you want to run a train or a test experiment.")
	parser.add_argument('--seeds', nargs='*', type=int, default=[0], help="RNG seeds to use for multiple iterations of the experiment.")
	parser.add_argument('--dataseeds', nargs='*', type=int, default=[100], help="RNG seeds to use for data preparation for multiple iterations of the experiment.")
	args = parser.parse_args()

	run_experiment(args.name, args.dataset, args.mode, args.seeds, args.dataseeds)



