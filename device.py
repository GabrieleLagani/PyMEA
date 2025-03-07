import os
import sys
import platform
import subprocess
import threading
from time import time, sleep
from struct import unpack
import torch
from queue import Queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

from pythonnet import load
load("coreclr")
import clr
sys.path.append('./API/') # add api to path
dll_modules = ['3Brain.BioCamDriver', '3Brain.Common']
for m in dll_modules:
	clr.AddReference(m) # import .dll into python
# Import C# Objects into Python from namespaces
from _3Brain.BioCamDriver import BioCamPool, BioCamIOSignal, StimEndPoint, BioCamStimExternalEndPoint, StimTrainProtocol
from _3Brain.Common import ChCoord, RectangularStimPulse
from System import Console
#Console.WriteLine("Hello from C#")

import utils
from params import *


# Thread pool class for MEA data reading and processing
class MultiProcMEADeviceReader:
	_executor = None
	_jobs = []

	@staticmethod
	def initialize(max_workers=1):
		if MultiProcMEADeviceReader._executor is None:
			MultiProcMEADeviceReader._executor = ProcessPoolExecutor(max_workers=max_workers)

	@staticmethod
	def shutdown():
		if MultiProcMEADeviceReader._executor is not None:
			MultiProcMEADeviceReader._executor.shutdown(wait=False)
		MultiProcMEADeviceReader._executor = None

	@staticmethod
	def submit(d):
		#byte_seq = bytes(d)
		future = MultiProcMEADeviceReader._executor.submit(MultiProcMEADeviceReader._worker_fn, d)
		MultiProcMEADeviceReader._jobs.append(future)

	@staticmethod
	def _worker_fn(d):
		d = utils.bytes2int(d)
		d = utils.digital2analog(d)
		# d = self.bioCam.Settings.RawStreamMask.GetMaskedInt16(d, 0, self.bioCam.DataFormat.DataPacketNFrames)
		# d = self.bioCam.DataFormat.DigitalToAnalog(d)
		# d = torch.tensor(d, dtype=torch.float16)
		#Device.data.append(d.reshape(-1, nh, nw))
		#Device.data.append(d.reshape(nh, nw, -1).permute(2, 0, 1))
		return d.reshape(-1, nh, nw)

	@staticmethod
	def wait_until_done():
		wait(MultiProcMEADeviceReader._jobs)

	@staticmethod
	def get_results():
		return [future.result() for future in MultiProcMEADeviceReader._jobs]

	@staticmethod
	def clear_jobs():
		MultiProcMEADeviceReader._jobs.clear()


# Class for accessing and interacting with MEA device
class MEADevice:
	record_until = 0
	packets = Queue()
	data_received_count = 0
	data_error_count = 0
	data_loss_count = 0
	stim_end_time = 0
	
	def __init__(self):
		MEADevice.log("Initializing MEA device...", create=True)
		self.bioCam = None
		self.isStreaming = False
		self.training = False
		self.true_fps = None

		self._wellIdx = None
		self._positiveEndPoints = None
		self._negativeEndPoints = None
		self._pulse = None
		self._protocol = None

		self._monitorshell = None

		def _listenproc():
			self._monitorshell = subprocess.Popen(
				"start pwsh /c Get-Content .devstate/output.log -Wait" if platform.system() == 'Windows' else "gnome-terminal --command=\"tail -f output.log\"",
				shell=True)
		
		self._listenthread = threading.Thread(name="Listen", target=_listenproc, args=())
		self._listenthread.start()

		def _readproc():
			MultiProcMEADeviceReader.initialize(max_workers=10)
			while True:
				d = MEADevice.packets.get()
				if d == 'done': break
				MultiProcMEADeviceReader.submit(d)
			MultiProcMEADeviceReader.shutdown()

		self._readthread = threading.Thread(name="Read", target=_readproc, args=())

	def train(self):
		self.training = True

	def eval(self):
		self.training = False

	@staticmethod
	def log(text, end='\n', create=False):
		os.makedirs('.devstate', exist_ok=True)
		with open('.devstate/output.log', 'w+' if create else 'a+') as f:
			f.write("[" + str(utils.tic()) + "] ")
			f.write(text)
			f.write(end)
			f.flush()

	@staticmethod
	def is_recording():
		return utils.tic() < MEADevice.record_until
	
	@staticmethod
	def set_record_until(until):
		MEADevice.record_until = until
		MEADevice.log("Recording until {}".format(until))
	
	@staticmethod
	def get_record_until():
		return MEADevice.record_until

	@staticmethod
	def reset_stim_end_time():
		MEADevice.stim_end_time = 0

	@staticmethod
	def get_stim_end_time():
		return MEADevice.stim_end_time
	
	# Event Handlers
	@staticmethod
	def BioCamPool_BioCamPoolStatusChanged(sender, e):
		slotInfoList = BioCamPool.BioCamSlotInfo
		for sinfo in slotInfoList: MEADevice.log("{} {} is {} connected and plate is {} ready".format(sinfo.CommercialModel, sinfo.SerialNumberAndVersion, '' if sinfo.IsBioCamConnected else 'not', '' if sinfo.IsMeaPlateConnected else 'not'))
	@staticmethod
	def BioCam_DataReceived(sender, e):
		if VERBOSE >= 3: MEADevice.log("Data received: {} | {}".format(e.Header, e.Payload))
		elif VERBOSE >= 2: MEADevice.log("Data received")
		if MEADevice.is_recording():
			MEADevice.data_received_count += 1
			#MEADevice.packets.put(e.Payload)
			MEADevice.packets.put(bytes(e.Payload))
			if VERBOSE >= 2: MEADevice.log("Data recorded")

	@staticmethod
	def BioCam_DataStreamingError(sender, e):
		if MEADevice.is_recording():
			MEADevice.data_error_count += 1

	@staticmethod
	def BioCam_DataLossAsync(sender, e):
		if MEADevice.is_recording():
			MEADevice.data_loss_count += 1

	@staticmethod
	def BioCam_ProtocolStatusChanged(sender, status):
		#print(status.Elapsed, status.Progress, status.Total)
		finished_time = utils.tic()
		if MEADevice.stim_end_time <= 0: MEADevice.stim_end_time = finished_time

	def get_true_fps(self):
		return self.bioCam.DataFormat.FrameRate

	def normalize_current(self, n, amplitude, width):
		#return amplitude, width * n
		#return amplitude * (n**0.5), width * (n**0.5)
		return amplitude * n, width

	def get_pulse(self, normalize=1):
		amplitude, width = (TET_STIM_AMPLITUDE, TET_STIM_WIDTH) if self.training else (STIM_AMPLITUDE, STIM_WIDTH)
		amplitude, width = self.normalize_current(normalize, amplitude, width)
		self._pulse = RectangularStimPulse('pulse', RectangularStimPulse.Default.Properties,
            #amplitude, width, 0, 0, 0)
			#amplitude, width, 0, amplitude, width)
			amplitude, width, 0, -amplitude, width)
		return self._pulse

	def send_signal(self, signal):
		if VERBOSE >= 1: MEADevice.log("Sending pulse with {} endpoint pairs".format(len(torch.nonzero(signal == 1))))

		# Select positive and negative endpoints
		self._positiveEndPoints = [self.bioCam.Stimulator.GetInternalEndPoint(ChCoord(self._wellIdx, l[0].item() + 1, l[1].item() + 1)) for l in torch.nonzero(signal == 1)]
		self._negativeEndPoints = [self.bioCam.Stimulator.GetInternalEndPoint(ChCoord(self._wellIdx, l[0].item() + 1, l[1].item() + 1)) for l in torch.nonzero(signal == -1)]

		# If no positive endpoint is selected, use external electrode as positive
		if len(self._positiveEndPoints) == 0: self._positiveEndPoints = [self.bioCam.Stimulator.GetExternalEndPoint(BioCamStimExternalEndPoint.Ext1Plus)]
		# If no negative endpoint is selected, use external electrode as negative
		if len(self._negativeEndPoints) == 0: self._negativeEndPoints = [self.bioCam.Stimulator.GetExternalEndPoint(BioCamStimExternalEndPoint.Ext1Minus)]

		# Send stimulus
		self.bioCam.Stimulator.Send(self.get_pulse(normalize=max(len(self._positiveEndPoints), len(self._negativeEndPoints))),
									self._positiveEndPoints, self._negativeEndPoints)
		
		# After a selected delay, disconnect activated electrodes to avoid electrical artifacts
		sleep(disconnectDelay / 1000)
		self.bioCam.ComPortSend(0x009C)

	def send_burst(self, signal, freq=10, duration=100):
		if VERBOSE >= 1: MEADevice.log("Sending burst with {} endpoint pairs".format(len(torch.nonzero(signal == 1))))

		# Select positive and negative endpoints
		self._positiveEndPoints = [self.bioCam.Stimulator.GetInternalEndPoint(ChCoord(l[0].item()+1, l[1].item()+1)) for l in torch.nonzero(signal == 1)]
		self._negativeEndPoints = [self.bioCam.Stimulator.GetInternalEndPoint(ChCoord(l[0].item()+1, l[1].item()+1)) for l in torch.nonzero(signal == -1)]

		# If no positive endpoint is selected, use external electrode as positive
		if len(self._positiveEndPoints) == 0: self._positiveEndPoints = [self.bioCam.Stimulator.GetExternalEndPoint(BioCamStimExternalEndPoint.Ext1Plus)]
		# If no negative endpoint is selected, use external electrode as negative
		if len(self._negativeEndPoints) == 0: self._negativeEndPoints = [self.bioCam.Stimulator.GetExternalEndPoint(BioCamStimExternalEndPoint.Ext1Minus)]

		# Send stimulus
		self._protocol = StimTrainProtocol('burst', self.get_pulse(normalize=max(len(self._positiveEndPoints), len(self._negativeEndPoints))),
									 RectangularStimPulse.Default.Properties, int(freq*duration/1000), float(freq))
		self._protocol.WellsIndexes = [self._wellIdx]
		self._protocol.PositiveEndPoints = self._positiveEndPoints
		self._protocol.NegativeEndPoints = self._negativeEndPoints
		self.bioCam.Stimulator.Protocol.LoadProtocol(0, self._protocol)
		self.bioCam.Stimulator.Protocol.StartProtocol(0)
		
		# After a selected delay, disconnect activated electrodes to avoid electrical artifacts
		utils.wait_until(utils.tic() + (disconnectDelay / 1000))
		self.bioCam.ComPortSend(0x009C)
	
	def read_activity(self):
		MEADevice.log("Reading recorded activity")
		#MultiProcMEADeviceReader.wait_until_done()
		raw = MultiProcMEADeviceReader.get_results()
		MEADevice.log("Read {} packets with shape {}".format(len(raw), raw[0].shape if len(raw) > 0 else None))
		series_raw = torch.cat(raw, dim=0) if len(raw) > 0 else torch.zeros([1, nh, nw], dtype=torch.float16)
		MultiProcMEADeviceReader.clear_jobs()
		return series_raw.reshape(-1, nh, nw)
	
	def __enter__(self):
		# Activate BioCam
		BioCamPool.Activate()
		BioCamPool.BioCamsStatusChanged += MEADevice.BioCamPool_BioCamPoolStatusChanged
		while not BioCamPool.IsActive: sleep(0.5)
		slotInfo = BioCamPool.BioCamSlotInfo[0]
		while not slotInfo.IsBioCamConnected: sleep(0.5)
		while not slotInfo.IsMeaPlateConnected: sleep(0.5)
		
		# Take BioCam control
		self.bioCam = BioCamPool.TakeBioCamControl(0)
		if not self.bioCam.IsConnected:
			raise RuntimeError("BioCam not connected")
		if not self.bioCam.MeaPlate.IsConnected:
			raise RuntimeError("MEA plate not connected")
		self._wellIdx = self.bioCam.MeaPlate.Settings[0].WellIdx
	
		# Register event handlers
		self.bioCam.DataReceived += MEADevice.BioCam_DataReceived
		if DEBUG: self.bioCam.DataStreamingError += MEADevice.BioCam_DataStreamingError
		if DEBUG: self.bioCam.DataLossAsync += MEADevice.BioCam_DataLossAsync
		
		# Set chamber temperature
		self.bioCam.MeaPlate.Settings.IsChamberTemperatureControlOn = True
		self.bioCam.MeaPlate.Settings.SetChamberTemperatureCelsius = 37.

		# Set hw filters
		self.bioCam.Settings.IsHpPostEnabled = HP_FILTER_FREQ is not None
		self.bioCam.Settings.HpPostCutOffFrequency = HP_FILTER_FREQ
		self.bioCam.Settings.HpPostOrder = HP_FILTER_ORDER
		self.bioCam.Settings.IsLpEnabled = LP_FILTER_FREQ is not None
		self.bioCam.Settings.LpCutOffFrequency = LP_FILTER_FREQ

		# Set internal chip calibration
		if calibrationBlanking is not None: self.bioCam.MeaPlate.CalibrationBlankingValue = calibrationBlanking
		if calibrationInterval is not None: self.bioCam.MeaPlate.CalibrationIntervalMs = calibrationInterval
		if amplifBias is not None:
			if amplifBias == 'auto':
				pass
				#self.bioCam.MeaPlate.StartAmplifiersBiasingCalibration()
				#sleep(3)
				#MEADevice.log("Amplifier bias calibrated to {}mV".format(self.bioCam.MeaPlate.Settings.AmplifiersBiasMilliVolt))
			else:
				self.bioCam.MeaPlate.SetAmplifiersBiasSoftAsync(amplifBias)
				sleep(3)
				MEADevice.log("Amplifier bias calibrated to {}mV".format(self.bioCam.MeaPlate.Settings.AmplifiersBiasMilliVolt))

		# Set sampling frequency
		MEADevice.log("Available frame rates {}Hz".format(list(self.bioCam.Settings.AproxTargetFrameRates)))
		MEADevice.log("Setting frame rate to {}Hz".format(samplingRate))
		if samplingRate not in list(self.bioCam.Settings.AproxTargetFrameRates):
			raise RuntimeError("Requested sampling rate {} does not match the available sampling rates {}".format(samplingRate, list(self.bioCam.Settings.AproxTargetFrameRates)))
		self.bioCam.Settings.AproxTargetFrameRate = samplingRate
		self.true_fps = self.get_true_fps()
		MEADevice.log("Frame rate set to {}Hz".format(self.true_fps))

		# Setting Calibration signal on pin 1, 1, and IO Signal on pin 1, 2
		self.bioCam.Settings.IOSignalsSettings[BioCamIOSignal.InternalChipCalibration].IsOn = True
		self.bioCam.Settings.IOSignalsSettings[BioCamIOSignal.InternalChipCalibration].OutputCh = ChCoord(1, 1)
		self.bioCam.Settings.IOSignalsSettings[BioCamIOSignal.InternalStimulation].IsOn = True
		self.bioCam.Settings.IOSignalsSettings[BioCamIOSignal.InternalStimulation].OutputCh = ChCoord(1, 2)
		
		# Start BioCam acquisition
		self.isStreaming = self.bioCam.StartDataStreaming(dataPacketTimeSpanMs=acquisitionTimePeriod, optimizedDataPacketLatency=True)
		if not self.isStreaming:
			raise RuntimeError("BioCam streaming error")
		self._readthread.start()
	
		# Configure stimulation on BioCam
		self.bioCam.Stimulator.Initialize()
		if amplifShutOff is not None:
			self.bioCam.Stimulator.Settings.IsStimCalibrationOn = True
			self.bioCam.Stimulator.Settings.CalibrationDistanceMicroSec = amplifShutOff
		if STIM_MODE == 'burst':
			self.bioCam.Stimulator.Protocol.InitializeProtocols(1)
			#self.bioCam.Stimulator.Protocol.PlayingProtocolProgressChanged += MEADevice.BioCam_ProtocolStatusChanged
		self.bioCam.Stimulator.Start()
		
		# Send calibration signal
		calibration = torch.zeros([nh, nw], dtype=torch.float32)
		calibration[nh//2:(nh//2)+1, nw//2] = 1.
		calibration[nh//2:(nh//2)+1, (nw//2)+1] = -1.
		self.send_signal(calibration)
		if amplifBias == 'auto':
			self.bioCam.MeaPlate.StartAmplifiersBiasingCalibration()
			sleep(3)
			MEADevice.log("Amplifier bias calibrated to {}mV".format(self.bioCam.MeaPlate.Settings.AmplifiersBiasMilliVolt))

		MEADevice.log("Settings are valid: {}".format(self.bioCam.MeaPlate.Settings.IsValid))
		MEADevice.log("MEA device ready!")
		print("MEA device ready")
		
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		MEADevice.log("Releasing device...")
		if self.bioCam is not None:
			# Terminate acquisition
			self.bioCam.Stimulator.Close()
			if not self.bioCam.StopDataStreaming():
				raise RuntimeError("Error while stopping BioCam acquisition")
			self.isStreaming = False
			self.bioCam.DataReceived -= MEADevice.BioCam_DataReceived
			if DEBUG: self.bioCam.DataStreamingError -= MEADevice.BioCam_DataStreamingError
			if DEBUG: self.bioCam.DataLossAsync -= MEADevice.BioCam_DataLossAsync
			BioCamPool.ReleaseBioCamControl(0)
			self.bioCam = None
		MEADevice.log("Device released!")
		MEADevice.log("You can close this window now")
		MultiProcMEADeviceReader.shutdown()
		self._monitorshell.kill()
		self._listenthread.join()
		self.packets.put('done')
		self._readthread.join()


# Device class for MEA interaction with optogenetic stimulation
class OptoMEADevice(MEADevice):

	def send_signal(self, signal):
		raise NotImplemented

	def send_burst(self, signal, freq=10, duration=100):
		raise NotImplemented
	

# Simulated device class used for debugging purposes. Does not require access to MEA
class DummyDevice:
	record_until = 0
	packets = Queue()
	data_received_count = 0
	data_error_count = 0
	data_loss_count = 0
	stim_end_time = 0
	
	def __init__(self):
		DummyDevice.log("Initializing dummy device...", create=True)
		self.bioCam = None
		self.isStreaming = False
		self.true_fps = None

		self._monitorshell = None
		
		# Load device state if available
		if os.path.exists('.devstate/dummy.dev'):
			DummyDevice.log("Loading device state...")
			self.load_state_dict(utils.load_dict('.devstate/dummy.dev'))
			DummyDevice.log("Device state loaded!")
			DummyDevice.log("Device state: {}".format(self.state_dict()))
		
		def _listenproc():
			self._monitorshell = subprocess.Popen(
				"start pwsh /c Get-Content .devstate/output.log -Wait" if platform.system() == 'Windows' else "gnome-terminal --command=\"tail -f output.log\"",
				shell=True)
		
		self._listenthread = threading.Thread(name="Listen", target=_listenproc, args=())
		self._listenthread.start()
	
	def state_dict(self):
		return {
			'bioCam': self.bioCam,
			'isStreaming': self.isStreaming,
			'training': self.training,
		}
	
	def load_state_dict(self, state_dict):
		self.bioCam = state_dict['bioCam']
		self.isStreaming = state_dict['isStreaming']
		if state_dict['training']: self.train()

	def train(self):
		self.training = True

	def eval(self):
		self.training = False

	@staticmethod
	def log(text, end='\n', create=False):
		os.makedirs('.devstate', exist_ok=True)
		with open('.devstate/output.log', 'w+' if create else 'a+') as f:
			f.write("[" + str(utils.tic()) + "] ")
			f.write(text)
			f.write(end)
			f.flush()
			
	@staticmethod
	def is_recording():
		return utils.tic() < DummyDevice.get_record_until()
	
	@staticmethod
	def set_record_until(until):
		DummyDevice.record_until = until
		DummyDevice.log("Recording until {}".format(until))
		
	@staticmethod
	def get_record_until():
		return DummyDevice.record_until

	@staticmethod
	def reset_stim_end_time():
		MEADevice.stim_end_time = 0

	@staticmethod
	def get_stim_end_time():
		return MEADevice.stim_end_time
	
	# Event Handlers
	@staticmethod
	def DataReceived(d):
		if VERBOSE >= 3: DummyDevice.log("Data received: {}".format(d))
		elif VERBOSE >= 2: DummyDevice.log("Data received")
		if DummyDevice.is_recording():
			DummyDevice.data_received_count += 1
			DummyDevice.packets.append(d)
			if VERBOSE >= 2: DummyDevice.log("Data recorded")

	def get_true_fps(self):
		return samplingRate

	def normalize_current(self, n, amplitude, width):
		#return amplitude, width * n
		#return amplitude * (n**0.5), width * (n**0.5)
		return amplitude * n, width

	def get_pulse(self):
		return None

	def send_signal(self, signal):
		if VERBOSE >= 1: DummyDevice.log("Sending pulse")
		sleep(1e-3)

	def send_burst(self, signal, freq=100, duration=100):
		if VERBOSE >= 1: DummyDevice.log("Sending signal")
		sleep(1e-3)

	def read_activity(self):
		DummyDevice.log("Reading recorded activity")
		series = DummyDevice.packets
		series_raw = torch.cat(series, dim=0) if len(series) > 0 else torch.zeros([1, nh, nw], dtype=torch.float16)
		DummyDevice.packets.clear()
		return series_raw.reshape(-1, nh, nw)
	
	def __enter__(self):
		def _dataproc():
			last_packet_time = 0
			while self.isStreaming:
				utils.wait_until(last_packet_time + acquisitionTimePeriod/1000)
				last_packet_time = utils.tic()
				DummyDevice.DataReceived(torch.randn([int(samplingRate*acquisitionTimePeriod/1000), 4096], dtype=torch.float16))
		
		# Check device availability
		self.log("Device available: {}".format(self.bioCam is None))
		while not self.bioCam is None: sleep(0.5)
		
		# Set state variables
		self.bioCam = 'dummy'
		self.true_fps = self.get_true_fps()
		self.isStreaming = True
		utils.save_dict(self.state_dict(), '.devstate/dummy.dev')
		
		# Start thread to simulate data arrival process
		self.datathread = threading.Thread(name="Data", target=_dataproc, args=())
		self.datathread.start()
		
		# Send calibration signal
		calibration = torch.zeros([nh, nw], dtype=torch.float32)
		calibration[nh // 2:(nh // 2) + 1, nw // 2] = 1.
		calibration[nh // 2:(nh // 2) + 1, (nw // 2) + 1] = -1.
		self.send_signal(calibration)
		
		DummyDevice.log("Dummy device ready!")
		print("Dummy device ready")
		
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		DummyDevice.log("Releasing device...")
		if self.bioCam is not None:
			self.bioCam = None
			self.isStreaming = False
			self.datathread.join()
		utils.save_dict(self.state_dict(), '.devstate/dummy.dev')
		DummyDevice.log("Device released!")
		DummyDevice.log("You can close this window now")
		self._monitorshell.kill()
		self._listenthread.join()
	

