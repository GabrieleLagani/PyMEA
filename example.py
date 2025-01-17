from time import time, sleep
from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
import sys

from pythonnet import load
load("coreclr")
import clr
sys.path.append('./API/') # add api to path
dll_modules = ['3Brain.BioCamDriver', '3Brain.Common']
for m in dll_modules:
	clr.AddReference(m) # import .dll into python
# Import C# Objects into Python from namespaces
from _3Brain.BioCamDriver import BioCamPool, StimEndPoint, BioCamStimExternalEndPoint, StimTrainProtocol
from _3Brain.Common import ChCoord, RectangularStimPulse
from System import Console
Console.WriteLine("Hello from C#")


# Experiment parameters
samplingRate = 6000.0 # Sampling frequency of data frames from MEA electrodes
acquisitionTimePeriod = 100 # Period in ms to record data from BioCam
stimPeriod = 1000 # Period in ms to deliver stimulations to BioCam
duration = 2 # Experiment duration in s
stimAmplitude = 40. # Stimulus pulse amplitude in uA
stimWidth = 20 # Stimulus pulse duration in us
pos_idx, neg_idx = (17, 8), (17, 10) # Stimulus coordinates

# Experiment state variables
data_received_count = 0
data = []
data_error_count = 0
data_loss_count = 0

# Event Handlers
def BioCamPool_BioCamPoolStatusChanged(sender, e):
	slotInfoList = BioCamPool.BioCamSlotInfo
	for sinfo in slotInfoList: print("{} {} is {} connected and plate is {} ready".format(sinfo.CommercialModel, sinfo.SerialNumberAndVersion, '' if sinfo.IsBioCamConnected else 'not', '' if sinfo.IsMeaPlateConnected else 'not'))
def BioCam_DataReceived(sender, e):
	print("Data received: {} | {}".format(e.Header, e.Payload))
	global data_received_count
	global data
	data_received_count += 1
	data.append(e.Payload)
def BioCam_DataStreamingError(sender, e):
	global data_error_count
	data_error_count += 1
def BioCam_DataLossAsync(sender, e):
	global data_loss_count
	data_loss_count += 1


startTime = time()
bioCam = None

try:
	# Activate BioCam
	BioCamPool.Activate()
	BioCamPool.BioCamsStatusChanged += BioCamPool_BioCamPoolStatusChanged
	while not BioCamPool.IsActive: sleep(0.5)
	slotInfo = BioCamPool.BioCamSlotInfo[0]
	while not slotInfo.IsBioCamConnected: sleep(0.5)
	while not slotInfo.IsMeaPlateConnected: sleep(0.5)
	print("{} {} is {} connected and plate is {} ready".format(slotInfo.CommercialModel, slotInfo.SerialNumberAndVersion, '' if slotInfo.IsBioCamConnected else 'not', '' if slotInfo.IsMeaPlateConnected else 'not'))

	# Take BioCam control
	bioCam = BioCamPool.TakeBioCamControl(0)
	if not bioCam.IsConnected:
		raise RuntimeError("BioCam not connected")
	if not bioCam.MeaPlate.IsConnected:
		raise RuntimeError("MEA plate not connected")
	print("BioCamPool attributes: {}".format(dir(BioCamPool)))
	print("bioCam attributes: {}".format(dir(bioCam)))
	print("Bytes per sample: {}".format(bioCam.NBytesPerChSample))
	print("Is Little endian: {}".format(bioCam.IsDataOnComPortLittleEndian))

	# Register event handlers
	bioCam.DataReceived += BioCam_DataReceived
	bioCam.DataStreamingError += BioCam_DataStreamingError
	bioCam.DataLossAsync += BioCam_DataLossAsync
	
	# Set chamber temperature
	bioCam.MeaPlate.Settings.IsChamberTemperatureControlOn = True
	bioCam.MeaPlate.Settings.SetChamberTemperatureCelsius = 37.
	
	# Set sampling frequency
	available_frame_rates = list(bioCam.Settings.AproxTargetFrameRates)
	print("Available frame rates: ")
	for afr in available_frame_rates:
		bioCam.Settings.AproxTargetFrameRate = afr
		print("approx {}, true {}".format(afr, bioCam.DataFormat.FrameRate))
	print("Setting frame rate to {}".format(samplingRate))
	bioCam.Settings.AproxTargetFrameRate = samplingRate
	print("True sampling rate set to {}".format(bioCam.DataFormat.FrameRate))

	# Start BioCam acquisition
	isStreaming = bioCam.StartDataStreaming(dataPacketTimeSpanMs=acquisitionTimePeriod, optimizedDataPacketLatency=True)
	if not isStreaming:
		raise RuntimeError("BioCam streaming error")

	# Configure stimulation on BioCam
	bioCam.Stimulator.Initialize()
	bioCam.Stimulator.Start()

	# Select positive and negative poles for stimulation
	positiveEndPoints = [bioCam.Stimulator.GetInternalEndPoint(ChCoord(*pos_idx))]
	negativeEndPoints = [bioCam.Stimulator.GetInternalEndPoint(ChCoord(*neg_idx))]
	
	# External end points
	positiveExtEndPoint = [bioCam.Stimulator.GetExternalEndPoint(BioCamStimExternalEndPoint.Ext1Plus)]
	NegativeExtEndPoint = [bioCam.Stimulator.GetExternalEndPoint(BioCamStimExternalEndPoint.Ext1Minus)]
	
	# Log stimulus information
	print("Stimulus properties: AmplitudeResolution {}, MinAmplitude {}, MaxAmplitude {}, TimeResolutionMicroSec {}".format(
			RectangularStimPulse.Default.Properties.AmplitudeResolution,
	        RectangularStimPulse.Default.Properties.MinAmplitude,
	        RectangularStimPulse.Default.Properties.MaxAmplitude,
	        RectangularStimPulse.Default.Properties.TimeResolutionMicroSec))

	sleep(stimPeriod // 1000)
	while time() < startTime + duration:
		# Send rectangular pulse on selected endpoints
		print("Sending pulse")
		bioCam.Stimulator.Send(
			#RectangularStimPulse.Default,
			RectangularStimPulse('pulse', RectangularStimPulse.Default.Properties, stimAmplitude, stimWidth, 0, 0, 0),
			positiveEndPoints, negativeEndPoints)

		sleep(stimPeriod // 1000)

finally:
	if bioCam is not None:
		# Terminate acquisition
		bioCam.Stimulator.Close()
		if not bioCam.StopDataStreaming():
			raise RuntimeError("Error while stopping BioCam acquisition")
		isStreaming = False
		bioCam.DataReceived -= BioCam_DataReceived
		bioCam.DataStreamingError -= BioCam_DataStreamingError
		bioCam.DataLossAsync -= BioCam_DataLossAsync
	BioCamPool.ReleaseBioCamControl(0)
	bioCam = None

	# Print results
	print("Packets received: {}, Errors received: {}, Packets lost: {}".format(data_received_count, data_error_count, data_loss_count))
	loc = (17, 9) # Location where to observe activity
	t = [1000, 3000] # Frame interval in which to observe activity
	series = np.array([], dtype='int16')
	for d in data:
		d = bytes(d)
		d = unpack('<' + 'h' * (len(d) // 2), d)
		d = np.array(d, dtype='int16')
		d = d[loc[0] * 64 + loc[1]::4096]
		series = np.concatenate([series, d], axis=0)
	print(len(series))
	plt.plot(series[t[0]:t[1]])
	plt.show()

