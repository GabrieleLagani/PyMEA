import json
import random
import os
from time import time, sleep
from struct import unpack, pack
import timeit
import csv
import h5py
import numpy as np
from scipy.signal import butter, filtfilt
import torch
import torch.nn.functional as F

import params as P
import utils

#tic = time
tic = timeit.default_timer

# Set rng seed
def set_rng_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# Set rng state
def set_rng_state(state):
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	random.setstate(state['python_rng'])
	np.random.set_state(state['numpy_rng'])
	torch.set_rng_state(state['pytorch_rng'])
	torch.cuda.set_rng_state_all(state['pytorch_rng_cuda'])

# Get rng state
def get_rng_state():
	state = {}
	state['python_rng'] = random.getstate()
	state['numpy_rng'] = np.random.get_state()
	state['pytorch_rng'] = torch.get_rng_state()
	state['pytorch_rng_cuda'] = torch.cuda.get_rng_state_all()
	return state

# Return formatted string with time information
def format_time(seconds):
	seconds = int(seconds)
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"

# Transforms shape tuple to size by multiplying the shape values
def shape2size(shape):
	size = 1
	for s in shape: size *= s
	return size

# Save state dictionary file to specified path
def save_dict(state_dict, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(state_dict, path)
	
# Load state dictionary file from specified path
def load_dict(path, device='cpu'):
	return torch.load(path, map_location=device)

# Save data to csv file
def update_csv(results, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, mode='w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		for name, entries in results.items():
			writer.writerow([name + '_epoch'] + list(entries.keys()))
			writer.writerow([name] + list(entries.values()))

# Convert MEA coordinates to linear index
def coords2idx(coords):
	return coords[0] * P.nw + coords[1]

# Convert MEA index to coordinates
def idx2coords(idx):
	return idx // P.nw, idx % P.nw

# Return the real MEA frame rate corresponding to the selected approximate frame rate
def fps_to_true_fps(fps):
	_fps_to_true_fps = {
		20000.0: 19753.775390625,
		18000.0: 18175.4453125,
		16000.0: 15671.1865234375,
		14000.0: 13773.4453125,
		12000.0: 11656.150390625,
		10000.0: 9876.8876953125,
		8000.0: 7574.67822265625,
		6000.0: 5543.99560546875,
	}
	return _fps_to_true_fps[fps]

def butter_highpass(cutoff, freq, order=2):
	nyq = 0.5 * freq
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	return b, a

b, a = None, None
def butter_highpass_filter(data, freq, cutoff=100, order=2):
	global b, a
	if b is None or a is None: b, a = butter_highpass(cutoff, freq, order=order)
	y = filtfilt(b, a, data.numpy(), axis=0)
	return torch.tensor(y.copy(), dtype=data.dtype)

# Spike detection based on sliding window
def detect_spikes(data, freq, window, sigma_thr, save_width):
	window = int(window * freq * 1e-3)
	save_width = int(save_width * freq * 1e-3)
	data = data.reshape(data.shape[0], -1)
	sigma = data.std(dim=0, keepdim=True)
	num_windows = data.shape[0] // window
	binned_data = data[:num_windows*window].reshape(num_windows, window, data.shape[1])
	pos_peak, neg_peak = binned_data.max(dim=1), binned_data.min(dim=1)
	idx = ((pos_peak[0] - neg_peak[0]) / sigma) > sigma_thr
	spike_times, channels, spike_forms, sf_starts = [], [], [], []
	for w in range(num_windows):
		centers = (neg_peak[1][w, idx[w]] + w*window).reshape(-1)
		st = centers.tolist()
		ch = torch.nonzero(idx[w]).reshape(-1).tolist()
		start, end = max(w*window-save_width, 0), min(w*window+save_width, data.shape[0])
		sf = list(data[start:end, idx[w]].transpose(1, 0))
		st, ch, sf = (l for l in zip(*sorted(zip(st, ch, sf), key=lambda x: x[0]))) if len(st) > 0 else (st, ch, sf)
		spike_times += st
		channels += ch
		spike_forms += sf
		sf_starts += [start] * len(sf)
	sigma_noise = sigma.reshape(-1).tolist()
	return spike_times, channels, spike_forms, sf_starts, sigma_noise

# Save recordings obtained from MEA to disk
def save_recording(raw, processed, pred, global_response, fps, stim_time, duration, dish_id, index, input, label, path):
	d = {
		'raw': raw,
		'processed': processed,
		'pred': pred,
		'global_response': global_response,
		'fps': fps,
		'stim_time': stim_time,
		'duration': duration,
		'dish_id': dish_id,
		'index': index,
		'input': input,
		'label': label,
	}
	save_dict(d, path)

# Save data permutation as json file
def save_data_permutation(path, trn_perm, tst_perm):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w+') as f:
		json.dump({'trn_perm': trn_perm, 'tst_perm': tst_perm}, f)

# Save recording params to disk
def save_recording_params(path):
	recording_params = {k: getattr(P, k) for k in dir(P) if not k.startswith('_')}
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w+') as f:
		json.dump(recording_params, f)

# Save recording data in the brw format
def save_recording_brw(processed, path):
	# Not implemented
	pass

# Convert analog to digital
def analog2digital(a, d_min=P.D_MIN, d_max=P.D_MAX, a_min=P.A_MIN, a_max=P.A_MAX):
	d = d_min + ((a - a_min) / (a_max - a_min)) * (d_max - d_min)
	d = d.to(dtype=torch.int16)
	return d

# Convert digital to analog
def digital2analog(d, d_min=P.D_MIN, d_max=P.D_MAX, a_min=P.A_MIN, a_max=P.A_MAX):
	d = d.to(dtype=torch.float16)
	a = a_min + ((d - d_min) / (d_max - d_min)) * (a_max - a_min)
	return a

def bytes2int(d):
	d = unpack(P.ENDIANNESS + 'h' * (len(d) // 2), d)
	return torch.tensor(d, dtype=torch.int16)

def int2bytes(t):
	d = pack(P.ENDIANNESS + 'h' * t.numel(), *(t.reshape(-1).tolist()))
	return d

# Sleep until specified time
def wait_until(end, eps=.1):
	while tic() < end:
		if end - tic() > eps: sleep(end - tic() - eps)
		else: sleep(eps / 20)

def gen_signal(input, label):
	padding= [1, 1, 1, 1]
	signal = F.interpolate(input.mean(dim=0).reshape(1, 1, input.shape[-2], input.shape[-1]), ((P.nh//2) - padding[0] - padding[1], (P.nw//2) - padding[2] - padding[3]))[0, 0, :, :]
	target = torch.zeros_like(signal)
	h_loc_0, w_loc_0 = (2 * (P.nh // 2)) // 3, (P.nw // 2) // 3
	h_loc_1, w_loc_1 = (2 * (P.nh // 2)) // 3, (2 * (P.nw // 2)) // 3
	target[h_loc_0-1:h_loc_0+1, w_loc_0-1:w_loc_0+1] = 1 if label == 0 else 0
	target[h_loc_1-1:h_loc_1+1, w_loc_1-1:w_loc_1+1] = 1 if label == 1 else 0
	signal, target = F.pad(signal, padding), F.pad(target, padding)
	masked_signal = torch.cat([signal, torch.zeros_like(target)], dim=-2)
	signal = torch.cat([signal, target], dim=-2)
	return signal, masked_signal
