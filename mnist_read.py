try:
	import urllib.request
except ImportError:
	raise ImportError(' use python version 3.x')
import os.path
import gzip, pickle, os
import numpy as np

url_base = "http://yann.lecun.com/exdb/mnist/"
file_name = {
	"train_img" : "train-images-idx3-ubyte.gz",
	"train_label" : "train-labels-idx1-ubyte.gz",
	"test_img" : "t10k-images-idx3-ubyte.gz",
	"test_label" : "t10k-labels-idx1-ubyte.gz"
}

current_root = os.path.dirname(os.path.abspath(__file__))
mnist_dir = current_root + "/mnist"
if not os.path.exists(mnist_dir):
	os.mkdir(mnist_dir)
save_file = mnist_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _download(file_name):
	file_path = os.path.join(mnist_dir, file_name)

	if os.path.exists(file_path):
		return

	print("download " + file_name + "...")
	urllib.request.urlretrieve(url_base + file_name, file_path)
	print(" download done")

def download_mnist():
	for v in file_name.values():
		_download(v)

def _load_data(file_name, type = None):
	file_path = os.path.join(mnist_dir, file_name)

	print(file_name + " to numpy array")

	if type is "img":
		#gzip
		with gzip.open(file_path, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=16)
			data = data.reshape(-1, img_size)
	elif type is "label":
		with gzip.open(file_path, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=8)
	print(" data converted")

	return data

def _convert_numpy():
	dataset = {}
	dataset['train_img'] = _load_data(file_name['train_img'], "img")
	dataset['train_label'] = _load_data(file_name['train_label'], "label")
	dataset['test_img'] = _load_data(file_name['test_img'], "img")
	dataset['test_label'] = _load_data(file_name['test_label'], "label")

	return dataset

def init_mnist():
	download_mnist()
	dataset = _convert_numpy()
	print("Creating pickle file")
	with open(save_file, 'wb') as f:
		pickle.dump(dataset, f, -1)
	print(" created pickle file")

def one_hot_coding(labels):
	temp = np.zeros((labels.size, 10))
	for idx, row in enumerate(temp):
		row[labels[idx]] = 1

	return temp

def load_mnist(normalize = True, flatten = False, one_hot = True):
	if not os.path.exists(save_file):
		init_mnist()
	with open(save_file, 'rb') as f:
		dataset = pickle.load(f)

	

	if normalize:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].astype(np.float32)
			dataset[key] = dataset[key]/255

	if one_hot:
		dataset['train_label'] = one_hot_coding(
			dataset['train_label'])
		dataset['test_label'] = one_hot_coding(
			dataset['test_label'])

	if not flatten:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].reshape(-1, 28, 28, 1)
	print(" current train_img shape : ", dataset['train_img'].shape)

	return dataset

if __name__ == '__main__':
	mnist = load_mnist()
