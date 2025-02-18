import os
import argparse
import math
import random
from tqdm import tqdm
from queue import Queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset

from data import CustomJsonDataset
from device import MEADevice, DummyDevice
import utils
from params import *


# Thread pool class for MEA output collection and post-processing
class MultiProcOutputCollector:
	_executor = None
	_jobs = []

	@staticmethod
	def initialize(max_workers=1):
		if MultiProcOutputCollector._executor is None:
			MultiProcOutputCollector._executor = ProcessPoolExecutor(max_workers=max_workers)

	@staticmethod
	def shutdown():
		if MultiProcOutputCollector._executor is not None:
			MultiProcOutputCollector._executor.shutdown(wait=False)
		MultiProcOutputCollector._executor = None

	@staticmethod
	def submit(o):
		future = MultiProcOutputCollector._executor.submit(MultiProcOutputCollector._worker_fn, o)
		MultiProcOutputCollector._jobs.append(future)

	@staticmethod
	def _worker_fn(o):
		raw, fs, mode, dish_id, index, input, label, assignments, delivery_latency, savepath = o
		raw, filtered, processed, pred, stim_time, global_response, delta = MultiProcOutputCollector.process_output(raw, fs, assignments)
		delivery_latency = (stim_time - RECORD_TIME[0]) if delivery_latency == 0 else delivery_latency
		if mode == 'test':
			utils.save_recording(raw, processed, pred, global_response, delta, fs, int(delivery_latency*fs + RECORD_TIME[0]*fs), int((RECORD_TIME[0]+RECORD_TIME[1])*fs), dish_id, index, input, label,
								 os.path.join(savepath, '{}'.format(label), '{}_{}.pt'.format(dish_id, index)))
			utils.save_recording_params(os.path.join(savepath, '{}'.format(label), '{}_{}_params.json'.format(dish_id, index)))
			utils.save_recording_brw(processed, os.path.join(savepath, '{}'.format(label), '{}_{}.brw'.format(dish_id, index)),
									 fs, int(delivery_latency*fs + RECORD_TIME[0]*fs), int((RECORD_TIME[0]+RECORD_TIME[1])*fs))
		print("\nRecording of sample {} completed with global response {} (delta {}) and latency {}".format(index, global_response, delta, delivery_latency))
		return pred, label, global_response, delta

	@staticmethod
	def process_output(raw, fs, assignments):
		filtered = raw #utils.butter_highpass_filter(raw, fs, HP_FILTER_FREQ, HP_FILTER_ORDER) if HP_FILTER_FREQ is not None else raw
		spike_times, channels, spike_forms, sf_starts, sigma_noise = utils.detect_spikes(filtered, fs, COMPR_WINDOW, COMPR_SIGMA_THR, COMPR_SAVE_WIDTH)
		processed = {'spike_times': spike_times, 'channels': channels, 'spike_forms': spike_forms, 'sf_starts': sf_starts, 'sigma_noise': sigma_noise}
		scores = MultiProcOutputCollector._scores_from_raw(filtered, assignments)
		#scores = MultiProcOutputCollector._scores_from_processed(processed, assignments)
		pred = torch.argmax(scores).item()
		# stim_time = RECORD_TIME[0]
		stim_time = min([st for i, st in enumerate(spike_times) if channels[i] == 1]) / fs
		global_response, delta =  MultiProcOutputCollector._global_response(spike_times, fs, stim_time, w_start=GLOBAL_RESPONSE_W_START, w_end=GLOBAL_RESPONSE_W_END)
		raw = raw if SAVE_RAW else None
		return raw, filtered, processed, pred, stim_time, global_response, delta

	@staticmethod
	def _scores_from_raw(raw, assignments):
		scores = [raw.reshape(raw.shape[0], -1)[:, assignments[c]].sum().item() for c in assignments]
		return torch.tensor(scores, dtype=torch.float16)

	@staticmethod
	def _scores_from_processed(processed, assignments):
		channels = processed['channels']
		scores = [len([ch for ch in channels if ch in assignments[c]]) for c in assignments]
		return torch.tensor(scores, dtype=torch.float16)

	@staticmethod
	def _global_response(spike_times, fs, ref_time, w_start=0, w_end=5e-3):
		ref_time, w_start, w_end = ref_time * fs, w_start * fs, w_end * fs
		res = len([st for st in spike_times if (st > ref_time + w_start) and (st <= ref_time + w_end)])
		delta = res - len([st for st in spike_times if (st < ref_time - w_start) and (st >= ref_time - w_end)])
		return res, delta

	@staticmethod
	def wait_until_done():
		wait(MultiProcOutputCollector._jobs)

	@staticmethod
	def get_results():
		return [future.result() for future in MultiProcOutputCollector._jobs]

	@staticmethod
	def clear_jobs():
		MultiProcOutputCollector._jobs.clear()

class Experiment:
	def __init__(self, exp_name, dish_id, dataset, mode, device, seed, dataseed, restart=False):
		print("Initializing experiment...")
		
		self.exp_name = exp_name
		self.dish_id = dish_id
		self.dataset = dataset
		self.mode = mode
		self.device = device
		self.seed = seed
		self.dataseed = dataseed
		self.data_folder = 'datasets'
		self.exp_folder = os.path.join('results', self.exp_name, 'iter{}'.format(self.seed))
		self.result_path = os.path.join(self.exp_folder, 'results.csv')
		self.checkpoint_path = os.path.join(self.exp_folder, 'checkpoint.pt')
		self.recordings_path = os.path.join(self.exp_folder, 'recordings')
		self.data_perm_save_path = os.path.join(self.exp_folder, 'dataperm.json')
		
		# Preparing data
		self.data_len = None
		self.electrode_assignments = None
		utils.set_rng_seed(self.dataseed)
		self.trn_set, self.tst_set = self.get_trn_tst_sets()
		
		# Initializing experiment state
		utils.set_rng_seed(self.seed)
		self.epoch = 0
		self.last_stim_time = 0
		self.outputs = Queue(maxsize=1)
		self.results = {'train_acc': {}, 'test_acc': {}}

		def _collectproc():
			MultiProcOutputCollector.initialize(max_workers=2)
			while True:
				o = self.outputs.get()
				if o == 'done': break
				MultiProcOutputCollector.submit(o)
			MultiProcOutputCollector.shutdown()

		self._collectthread = threading.Thread(name="Collect", target=_collectproc, args=())
		
		# Resuming from checkpoint, if necessary
		if (not restart and os.path.exists(self.checkpoint_path)) or self.mode == 'test':
			if self.mode == 'test' and not os.path.exists(self.checkpoint_path):
				print("Warning: no checkpoint found in path {}... Testing from scratch...".format(self.checkpoint_path))
			else:
				print("Loading experiment state from checkpoing {}...".format(self.checkpoint_path))
				self.load_state_dict(utils.load_dict(self.checkpoint_path))
				print("Checkpoint loaded!")
	
	def state_dict(self):
		return {
			'epoch': self.epoch + 1,
			'last_stim_time': self.last_stim_time,
			'rng_state': utils.get_rng_state(),
		}
	
	def load_state_dict(self, state_dict):
		self.epoch = state_dict['epoch']
		self.last_stim_time = state_dict['last_stim_time']
		utils.set_rng_state(state_dict['rng_state'])
	
	def save_results(self):
		# Save results to csv file
		utils.update_csv(self.results, self.result_path)
		
		# Save experiment state to checkpoint file
		utils.save_dict(self.state_dict(), self.checkpoint_path)

	def get_trn_tst_sets(self):
		dataset = CustomJsonDataset(os.path.join(self.data_folder, self.dataset), repeats=CUSTOM_DATASET_REPEATS)
		self.data_len = len(dataset)
		self.electrode_assignments = {c: dataset.get_electrode_assignments(c) for c in dataset.classes}
		utils.save_data_permutation(self.data_perm_save_path, dataset.sequence, dataset.sequence)
		return self._get_trn_tst_sets(dataset, dataset)

	def _get_trn_tst_sets(self, trn_dataset, tst_dataset):
		return DataLoader(trn_dataset, batch_size=1, shuffle=True, num_workers=1), DataLoader(tst_dataset, batch_size=1, shuffle=False, num_workers=1)

	def deliver_signal(self, signal, max_freq, duration):
		self.last_stim_time = utils.tic()
		T = int(max_freq * duration / 1000) if STIM_MODE == 'pulse' else 1
		bin_size = duration / T
		eps = (1 / max_freq)
		last_pulse_time = 0
		for t in range(0, T):
			spikes = signal
			# Send signal to device
			utils.wait_until(last_pulse_time + eps)
			last_pulse_time = utils.tic()
			if STIM_MODE == 'pulse':
				self.device.send_signal(spikes)
			elif STIM_MODE == 'burst':
				self.device.send_burst(spikes, max_freq, duration)
			else:
				raise ValueError("Unsupported stimulation mode {}, only pulse or burst available".format(STIM_MODE))
	
	def process_sample(self, index, input, target, label):
		# Encode input
		signal, masked_signal = input, input
		if target != '':
			signal = torch.clamp(input + target, min=-1, max=1)
			masked_signal = utils.mask_signal(signal, target)
		
		# Perform tetanization if training
		if self.mode == 'train':
			self.device.train()
			utils.wait_until(self.last_stim_time + STIM_INTERVAL)
			self.deliver_signal(signal, max_freq=TET_MAX_FREQ, duration=TET_DURATION)
			self.device.eval()
		
		# Start activity recording for next stimulation
		utils.wait_until(self.last_stim_time + STIM_INTERVAL - RECORD_TIME[0])
		self.device.set_record_until(utils.tic() + RECORD_TIME[0] + RECORD_TIME[1])
		record_started = utils.tic()
		
		# Send stimulation for evaluation and record activity
		utils.wait_until(record_started + RECORD_TIME[0])
		self.device.reset_stim_end_time()
		self.deliver_signal(masked_signal, max_freq=STIM_MAX_FREQ, duration=STIM_DURATION)
		setup_latency = utils.tic() - self.last_stim_time
		
		# Wait for device recording to complete, then read recorded activity
		utils.wait_until(self.device.get_record_until())
		raw = self.device.read_activity()
		delivery_latency = max(self.device.get_stim_end_time() - self.last_stim_time, 0)
		self.outputs.put((raw, self.device.true_fps, self.mode, self.dish_id, index, input, label, self.electrode_assignments, delivery_latency, self.recordings_path + '{}'.format(self.epoch)))
	
	def process_epoch(self, dataset):
		# Iterate through dataset, send inputs to device, record and process outputs
		for indexes, inputs, targets, labels in tqdm(dataset):
			for index, input, target, label in zip(indexes, inputs, targets, labels):
				self.process_sample(str(index.item()).zfill(int(math.log10(self.data_len))+1), input, target, label)

		# Once predictions have been collected, determine epoch performance
		hits, count = 0, 0
		predictions = MultiProcOutputCollector.get_results()
		for pred, label, _, _ in predictions:
			res = (pred == label).int().sum().item()
			hits += res
			count += 1
		MultiProcOutputCollector.clear_jobs()
		return hits / count
	
	def train(self):
		# Start parallel data collection thread
		self._collectthread.start()

		# Train loop
		for epoch in range(self.epoch, EPOCHS+1):
			start_time = utils.tic()
			
			self.epoch = epoch
			print("\nTRAIN | Epoch {}/{} | Experiment {} | Iter {}".format(self.epoch, EPOCHS, self.exp_name, self.seed))
			
			print("Training...")
			trn_acc = self.process_epoch(self.trn_set)
			print("Train Accuracy: {}".format(trn_acc))
			self.results['train_acc'][epoch] = trn_acc

			#print("Testing...")
			#tst_acc = self.process_epoch(self.tst_set)
			#self.results['test_acc'][epoch] = tst_acc
			
			self.save_results()
			
			epoch_duration = utils.tic() - start_time
			print("Epoch duration: " + utils.format_time(epoch_duration))
			print("Expected remaining time: " + utils.format_time((EPOCHS - epoch) * epoch_duration))

		# Terminate data collection thread
		self.outputs.put('done')
		self._collectthread.join()
	
	def eval(self):
		# Start parallel data collection thread
		self._collectthread.start()

		# Test
		print("EVAL | Experiment {} | Iter {}".format(self.exp_name, self.seed))
		print("Testing...")
		tst_acc = self.process_epoch(self.tst_set)
		print("Test Accuracy: {}".format(tst_acc))

		# Terminate data collection thread
		self.outputs.put('done')
		self._collectthread.join()


def run_experiment(name, dish_id, dataset, mode, device, seeds, dataseeds, restart):
	Device = DummyDevice
	if device == 'mea': Device = MEADevice
	
	for iter, seed in enumerate(seeds):
		print("\n####    Iter {}    ####".format(seed))
		dataseed = dataseeds[iter % len(dataseeds)]
		try:
			with Device() as d:
				if 'train' in mode: Experiment(name, dataset, dish_id, 'train', d, seed, dataseed, restart).train()
				if 'test' in mode: Experiment(name, dish_id, dataset, 'test', d, seed, dataseed, restart).eval()
				print("\nIter {} completed!".format(seed))
		except:
			raise
	print("\n\nFinished!")


if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help="Name assigned to the experiment")
	parser.add_argument('--dish_id', type=str, help="ID of the dish being used for the experiment")
	parser.add_argument('--dataset', default='mnist', type=str, help="Dataset name to be used for the experiment")
	parser.add_argument('--mode', default='test', choices=['train', 'test', 'traintest'], help="Whether you want to run a train or a test experiment.")
	parser.add_argument('--device', default='dummy', choices=['dummy', 'mea'], help="The device to use for the experiment.")
	parser.add_argument('--seeds', nargs='*', type=int, default=[0], help="RNG seeds to use for multiple iterations of the experiment.")
	parser.add_argument('--dataseeds', nargs='*', type=int, default=[100], help="RNG seeds to use for data preparation for multiple iterations of the experiment.")
	parser.add_argument('--restart', action='store_true', default=False, help="Whether you want to restart the experiment from scratch, overwriting previous checkpoints in the save path.")
	args = parser.parse_args()

	run_experiment(args.name, args.dish_id, args.dataset, args.mode, args.device, args.seeds, args.dataseeds, args.restart)



