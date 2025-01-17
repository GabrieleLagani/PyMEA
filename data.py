from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision
import torchvision.transforms as tt
from PIL import ImageFilter

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

