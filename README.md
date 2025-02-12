# PyMEA: Multi-Electrode Array Recording and Stimulation with Python
This repository contains a Python interface and example code to use Multi-Electrode Arrays (MEA)
devices for recording and stimulation. Specifically, this code has been developed to provide a 
simple interfacing with the [3Brain BioCam DupleX](https://www.3brain.com/products/single-well/biocam-duplex) 
MEA device, API v. 2.4.
The APIs for the device are originally writtent in C#. However, given the widespread use of the 
Python language for scientific research, we found it worthwhile to dedicate some effort in developing
a Python porting of the interface.

MEA devices allow scientists to interact with cultured networks of biological neurons, both in terms
of recording neural activity, and also for delivering stimulation patterns to networks and observing 
the responses. The 3Brain device offers a 64x64 grid of electrodes that can be used both for 
stimulation and recording.

Given this technology, our research group is investigating the opportunity to use cultured neural 
networks to address AI challenges: is it possible to harness the computing power of biological neurons
to perform complex pattern recognition tasks? And is it possible to leverage the synaptic 
plasticity mechanisms of such neurons to guide the behavior of the network towards a desired outcome?
These are some of the questions that our research group is trying to answer.

The ambitious goal of this project lies at the intersection of computer science and neuroscience.
Given the strong connections with the deep learning field, the need for a Python interface, that we provide 
in this repo, stems from the appealing perspective of being able to use the vast pool of deep learning 
libraries that are available in Python. Therefore, we deem that this contribution can represent a
valid tool for other researchers interested in this line of work.

As a use case example, the code provides an experimental scenario in which the Pytorch library is
used in conjunction with the MEA APIs. Specifically, through Pytorch we have a simple interface 
to access popular computer vision datasets for benchmarking, such as MNIST, CIFAR10, CIFAR100.
We can use the images contained in this dataset as a set of stimuli to be delivered to a cultured
network, through the MEA electrodes. Think of an electrode as corresponding to a pixel of an image.
The intensity of the pixel is mapped to a corresponding spike train on the electrode, with frequency
proportional to the pixel intensity. We can loop through the dataset images, and show them to the
cultured network, one by one. In the meantime we can record the activity of neurons in response to
these stimuli, and use them as a high-dimensional feature representation for downstream tasks, such as
classification. Furthermore, a tetanization protocol (i.e. a mixture of high frequency stimulations) 
can be defined, in order to induce plasticity in the neurons, and modify their input-output response
through learning.

## Contents

The file `device.py` contains the Python interface to the MEA APIs.

The file `main.py` is the main entry point of the stimulation protocol with image datasets.
You can run it with:

`python main.py`

The file `params.py` contains the various parameters to set up the MEA device and run the 
stimulation and recording.

The file `exp.py` contains the experiment logic. The implementation is optimized in performance and 
leverages multiprocessing and vector operations, because the MEA can provide data with a
very large frame rate, so it is essential to maximize efficiency to keep up.

The file `brw.py` provides utility functions for reading data from the BrainWave file format, and 
for converting saved recording dictionaries to such format.

The files `make_json_dataset.py` and `json_data_convert.py` provide another entry point which allows users to transform samples
from computer vision datasets to json files that can be read by the default MEA GUI interfacing tool 
provided by 3Brain. Once a given image is saved as a json file, it can be opened by the tool and used 
for stimulation and recording from the GUI. Of course this is useful to observe the cultured network 
in real time, but cannot scale to manually showing thousands of images.

Files `run_json_exp.py` and `json_exp.py` can be used to drive stimulation protocols with a custom JSON dataset containing 
stimulus patterns designed through the BrainWave interface. The custom dataset folder must be placed 
inside the `datasets` directory. The internal structure of the folder must be organized as follows:
a subfolder `samples` contains the JSON files; an additional optional folder `targets` can contain a selection of desired
target output electrodes, again encoded as JSON files. If targets are used, each target file must correspond to one sample 
file and vice-versa.


Files `utils.py`, `data.py`, and `visualize.py` provide other utility functions.

## Requirements

.Net Framework 4.7 is needed to run the MEA interface APIs. These C# APIs are then ported
to Python via `pythonnet`. The `.dll` files of the APIs are provided by 3Brain to its customers.
Additional settings might be necessary. Contact the Author in case of problems.
Python 3.10 or higher is recommended. Other necessary packages are listed in `requirements.txt`.

## Author

Gabriele Lagani: `gabriele.lagani@gmail.com`.



