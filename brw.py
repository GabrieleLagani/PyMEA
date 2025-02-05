import argparse
import h5py
import json
from struct import pack, unpack
import uuid
import numpy as np
import datetime

import torch

import utils


_PATH = r"C:\Users\BioCAM User\Desktop\PyMEA\results\mnist\iter0\recordings\0\N1_DIV30_25928.pt"
_FRAMES_PER_CHUNK = 24690
_SYNC_SIGNAL_PERIOD = 0.4 # None to disable. Default 400. Period in s of the synchronization signals to visualize on channel 0
_SYNC_SIGNAL_DELAY = 0.07 # None to disable. Default 0. Delay in s of the first synchronization signal to visualize on channel 0

MaxAnalogValue = 8000.0
MinAnalogValue = -8000.0
MaxDigitalValue = 4096 - 1
MinDigitalValue = 0

def _type_agnostic_da_convert(data, affine=True):
	offset = MinAnalogValue if affine else 0
	return offset + data * (MaxAnalogValue - MinAnalogValue) / (MaxDigitalValue - MinDigitalValue)

def _type_agnostic_ad_convert(data, affine=True):
	offset = MinAnalogValue if affine else 0
	return (data - offset) * (MaxDigitalValue - MinDigitalValue) / (MaxAnalogValue - MinAnalogValue)

# Read recordings from brw file, in the specified time interval. If channels are provided, read only the selected channels. Read all channels by default.
def read_brw(path, startFrame, numFrames, channels=None):
	if channels is None: channels = list(range(4096))
	data = {chIdx: {} for chIdx in channels}

	print("Opening file {}...".format(path))
	with h5py.File(path, "r") as f:
		# collect the TOCs
		toc = np.array(f['TOC'])
		eventsToc = np.array(f['Well_A1/EventsBasedSparseRawTOC'])

		# from the given start position and duration in frames, localize the corresponding event positions using the TOC
		tocStartIdx = np.searchsorted(toc[:, 1], startFrame)
		tocEndIdx = min(np.searchsorted(toc[:, 1], startFrame + numFrames, side='right') + 1, len(toc) - 1)
		eventsStartPosition = eventsToc[tocStartIdx]
		eventsEndPosition = eventsToc[tocEndIdx]

		# decode all data for the given well ID and time interval
		binaryData = f['Well_A1/EventsBasedSparseRaw'][eventsStartPosition:eventsEndPosition]
		binaryDataLength = len(binaryData)

		pos = 0
		while pos < binaryDataLength:
			print("\r\33[KReading... {:.2f}%".format(100 * pos / binaryDataLength), end="")
			chIdx = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
			pos += 4

			chDataLength = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
			pos += 4

			if chIdx not in channels:
				pos += chDataLength
				continue

			chDataPos = pos
			while pos < chDataPos + chDataLength:
				fromInclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
				pos += 8

				toExclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
				pos += 8

				data[chIdx][fromInclusive] = np.zeros(toExclusive - fromInclusive, dtype=np.float16)
				rangeDataPos = pos
				for j in range(fromInclusive, toExclusive):
					if j >= startFrame + numFrames:
						break
					if j >= startFrame:
						data[chIdx][fromInclusive][j - fromInclusive] = _type_agnostic_da_convert(int.from_bytes(binaryData[rangeDataPos:rangeDataPos + 2], byteorder='little', signed=True))

					rangeDataPos += 2

				pos += (toExclusive - fromInclusive) * 2
		print("\r\33[KReading... 100.00%")

	return data

def write_brw(path, spike_times, channels, spike_forms, sf_starts, sigma_noise, fs, stim_time, duration, frames_per_chunk=_FRAMES_PER_CHUNK):
	def _overwrite_sync_channel(spike_times, channels, spike_forms, sf_starts, sigma_noise, fs):
		out_spike_times, out_channels, out_spike_forms, out_sf_starts = [], [], [], []
		prev_s = -1
		for st, ch, sf, s in zip(spike_times, channels, spike_forms, sf_starts):
			if ch != 0:
				for t in range(prev_s - ((prev_s - int(_SYNC_SIGNAL_DELAY*fs)) % int(_SYNC_SIGNAL_PERIOD*fs)), st, int(_SYNC_SIGNAL_PERIOD*fs)):
					if t >= prev_s:
						out_spike_times.append(t)
						out_channels.append(0)
						out_spike_forms.append(torch.tensor([MaxAnalogValue if i == len(spike_forms[0])//2 else 0. for i in range(len(spike_forms[0]))], dtype=torch.float16))
						out_sf_starts.append(t)
				out_spike_times.append(st)
				out_channels.append(ch)
				out_spike_forms.append(sf)
				out_sf_starts.append(s)
				prev_s = s
		sigma_noise[0] = 0
		return out_spike_times, out_channels, out_spike_forms, out_sf_starts, sigma_noise

	def _overwrite_ref_channel(spike_times, channels, spike_forms, sf_starts, sigma_noise, stim_time):
		spike_forms = [sf for i, sf in enumerate(spike_forms) if channels[i] != 1]
		sf_starts = [s for i, s in enumerate(sf_starts) if channels[i] != 1]
		spike_times = [st for i, st in enumerate(spike_times) if channels[i] != 1]
		channels = [ch for i, ch in enumerate(channels) if channels[i] != 1]
		idx = np.searchsorted(spike_times, stim_time)
		spike_forms.insert(idx, torch.tensor([MaxAnalogValue if i == len(spike_forms[0])//2 else 0. for i in range(len(spike_forms[0]))], dtype=torch.float16))
		sf_starts.insert(idx, stim_time - len(spike_forms[0])//2)
		spike_times.insert(idx, stim_time)
		channels.insert(idx, 1)
		sigma_noise[1] = 0
		return spike_times, channels, spike_forms, sf_starts, sigma_noise

	def _write_range(start, samples):
		fromInclusive, toExclusive = start, start + len(samples)
		r = np.asarray(_type_agnostic_ad_convert(samples).astype(np.int16), dtype='<i2').tobytes()
		r = np.asarray([fromInclusive, toExclusive], dtype='<i8').tobytes() + r
		return r

	def _write_channel(channel, ranges):
		chIdx = np.array([channel], dtype='<i4').tobytes()
		encRanges = [_write_range(s, r) for s, r in ranges]
		chData = b''
		for r in encRanges:
			chData += r
		chDataLength = np.array([len(chData)], dtype='<i4').tobytes()
		chData = chIdx + chDataLength + chData
		return chData

	def _write_chunk(chunk):
		encChunk = b''
		for channel in chunk:
			encChunk += _write_channel(channel, chunk[channel])
		return encChunk

	def _write_sparse_raw(spike_times, channels, spike_forms, sf_starts, sigma_noise, fs, stim_time, duration, frames_per_chunk):
		spike_times, channels, spike_forms, sf_starts, sigma_noise = _overwrite_sync_channel(spike_times, channels, spike_forms, sf_starts, sigma_noise, fs)
		spike_times, channels, spike_forms, sf_starts, sigma_noise = _overwrite_ref_channel(spike_times, channels, spike_forms, sf_starts, sigma_noise, stim_time)
		sf_ends = [s + len(spike_forms[i]) for i, s in enumerate(sf_starts)]
		last_frame = max(sf_ends)
		data = b''
		toc = []
		eventToc = []
		noiseMean, noiseStdDev, noiseChIdxs, noiseToc = [], [], [], []
		for chunk_start in range(0, last_frame, frames_per_chunk):
			chunk_end = chunk_start + frames_per_chunk
			chunk = {chIdx: [] for chIdx in range(4096)}
			for j, s in enumerate(sf_starts):
				e = s + len(spike_forms[j])
				if s >= chunk_start and s < chunk_end:
					chunk[channels[j]].append( (s, np.asarray(spike_forms[j], dtype=np.float16)) )
			toc.append( (chunk_start, chunk_end) )
			eventToc.append(len(data))
			data += _write_chunk(chunk)
			noiseToc.append(len(noiseMean)*4)
			noiseMean += _type_agnostic_ad_convert(np.zeros_like(np.asarray(sigma_noise, dtype=np.float32))).tolist()
			noiseStdDev += _type_agnostic_ad_convert(np.asarray(sigma_noise, dtype=np.float32), affine=False).tolist()
			noiseChIdxs += list(range(len(sigma_noise)))
		return [b for b in data], eventToc, toc, noiseMean, noiseStdDev, noiseChIdxs, noiseToc

	EventsBasedSparseRaw, eventToc, toc, noiseMean, noiseStdDev, noiseChIdxs, noiseToc = _write_sparse_raw(spike_times, channels, spike_forms, sf_starts, sigma_noise, fs, stim_time, duration, frames_per_chunk=frames_per_chunk)

	with h5py.File(path, 'a') as of:
		of.attrs['Description'] = np.asarray(b'')
		of.attrs['ExperimentDateTimeUtc'] = int(datetime.datetime.now(datetime.UTC).timestamp())
		of.attrs['ExperimentType'] = 0
		of.attrs['GUID'] = np.asarray(bytes(str(uuid.uuid1()), encoding='utf-8')) #np.asarray(b'678fa061-2d4a-47ba-9d79-12bad2f897a1')
		of.attrs['MaxAnalogValue'] = 8000.0
		of.attrs['MaxDigitalValue'] = 4095.0
		of.attrs['MinAnalogValue'] = -8000.0
		of.attrs['MinDigitalValue'] = 0.0
		of.attrs['PlateModel'] = 12
		of.attrs['SamplingRate'] = fs
		of.attrs['Version'] = 400
		of.create_dataset('ExperimentSettings', data=np.asarray([b''], np.dtypes.ObjectDType))
		of['ExperimentSettings'].attrs['Status'] = 0
		of.create_dataset('ImageLayers', data=np.asarray([], dtype='<u1'))
		of.create_dataset('TOC', data=np.asarray(toc, dtype='<i8'))
		of.create_group("Well_A1")
		of['Well_A1'].attrs['Version'] = 100
		of["Well_A1"].create_dataset("EventsBasedSparseRaw", data=np.asarray(EventsBasedSparseRaw, dtype='<u1'))
		of["Well_A1"].create_dataset("EventsBasedSparseRawTOC", data=np.asarray(eventToc, dtype='<i8'))
		of["Well_A1"].create_dataset("NoiseChIdxs", data=np.asarray(noiseChIdxs, dtype='<i4'))
		of["Well_A1"].create_dataset("NoiseMean", data=np.asarray(noiseMean, dtype='<f4'))
		of["Well_A1"].create_dataset("NoiseStdDev", data=np.asarray(noiseStdDev, dtype='<f4'))
		of["Well_A1"].create_dataset("NoiseTOC", data=np.asarray(noiseToc, dtype='<i8'))
		of["Well_A1"].create_dataset("StoredChIdxs", data=np.asarray(list(range(4096)), dtype='<i4'))

def spikes_from_bxr(path):
	with h5py.File(path, "r") as f:
		w = f["Well_A1"]
		spike_times = np.array(w["SpikeTimes"])
		channels = np.array(w["SpikeChIdxs"])
		spike_forms = np.array(w["SpikeForms"])
		spike_forms = _type_agnostic_da_convert(spike_forms.reshape(len(spike_times), -1))
	return spike_times, channels, spike_forms

def pt2brw(path):
	# Load saved dictionary with keys: raw, processed, pred, global_response, fps, stim_time, dish_id, index, input, label, path
	d = utils.load_dict(path)
	data = d['processed']
	write_brw(path.replace('.pt', '.brw'), data['spike_times'], data['channels'], data['spike_forms'], data['sf_starts'], data['sigma_noise'], d['fps'], d['stim_time'], d['duration'])


if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default=_PATH, help="Path of the '.pt' file to be converted to '.brw'")

	args = parser.parse_args()

	pt2brw(args.path)

