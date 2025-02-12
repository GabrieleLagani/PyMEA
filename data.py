import os
import random
import json

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision
import torchvision.transforms as tt
from PIL import ImageFilter

import params as P
import utils


class CustomJsonDataset(Dataset):
	def __init__(self, path, repeats=1):
		self.path = os.path.join(path, 'samples')
		self.target_path = os.path.join(path, 'targets')
		if not os.path.exists(self.target_path): self.target_path = None
		self.files = [f for f in os.listdir(self.path) if (not os.path.isdir(os.path.join(self.path, f))) and f.endswith('.json')]
		self.files = sorted(self.files)
		if self.target_path is not None:
			self.target_files = [f for f in os.listdir(self.target_path) if not os.path.isdir(f) and f.endswith('.json')]
			self.target_files = sorted(self.target_files)
			unmatched_files = [f for i, f in enumerate(self.files) if f not in self.target_files]
			unmatched_targets = [m for i, m in enumerate(self.target_files) if m not in self.files]
			if len(unmatched_targets) + len(unmatched_files) > 0:
				raise RuntimeError("Inconsistent dataset format. Sample and masl files should match, but found unmatching files {} and/or targets {}".format(unmatched_files, unmatched_targets))
		self.classes = [i for i, _ in enumerate(self.files)]

		self.repeats = repeats
		self.sequence = None
		if isinstance(self.repeats, (list, tuple)):
			self.sequence = self.repeats
		elif isinstance(self.repeats, int):
			sequence = [i % len(self.files) for i in range(len(self.files) * self.repeats)]
			random.shuffle(sequence)
			self.sequence = sequence
		if self.sequence is None:
			raise ValueError("Expected repeats to be integer or list, but found {}".format(self.repeats))

		self.targets = self.sequence

	def read_file(self, file_path):
		pe, ne = [], []
		with open(file_path, 'r') as f:
			settings = json.load(f)
			groups = settings['Groups']
			for g in groups:
				if g['Name'] == 'PositiveEndpoints': pe += g['PixelIndexes']
				if g['Name'] == 'NegativeEndpoints': ne += g['PixelIndexes']
		return pe, ne

	def load_from_file(self, file_path):
		pe, ne = self.read_file(file_path)
		sample = torch.zeros(P.nh * P.nw, dtype=torch.float32)
		sample[pe] = 1.
		sample[ne] = -1.
		return sample.reshape(P.nh, P.nw)

	def __getitem__(self, idx):
		file_path = os.path.join(self.path, self.files[self.sequence[idx]])
		input = self.load_from_file(file_path)

		target = ''
		if self.target_path is not None:
			file_path = os.path.join(self.target_path, self.target_files[self.sequence[idx]])
			target = self.load_from_file(file_path)

		label = self.targets[idx]

		return idx, input, target, label

	def __len__(self):
		return len(self.sequence)

	def get_electrode_assignments(self, cls, pad=1):
		file_path = os.path.join(self.path, self.files[cls])
		if self.target_path is not None: file_path = os.path.join(self.target_path, self.target_files[cls])
		pe, ne = self.read_file(file_path)
		locs = []
		for l in pe + ne:
			y, x = utils.idx2coords(l)
			locs += [utils.coords2idx((y + p_y, x + p_x)) for p_y in range(-pad, pad+1) for p_x in range(-pad, pad+1)]
		return locs


class CustomMNIST(MNIST):
	def __getitem__(self, idx):
		input, label = super().__getitem__(idx)
		return idx + (0 if self.train else 60000), input, label

class CustomCIFAR10(CIFAR10):
	def __getitem__(self, idx):
		input, label = super().__getitem__(idx)
		return idx + (0 if self.train else 50000), input, label

class CustomCIFAR100(CIFAR100):
	def __getitem__(self, idx):
		input, label = super().__getitem__(idx)
		return idx + (0 if self.train else 50000), input, label


class Edgify:
	def __call__(self, image):
		image = image.convert("L")
		image = image.filter(ImageFilter.FIND_EDGES)
		return image

def get_transform(detect_edges=False):
	if detect_edges: return tt.Compose([tt.Grayscale(), Edgify(), tt.ToTensor()])
	return tt.Compose([tt.Grayscale(), tt.ToTensor()])

