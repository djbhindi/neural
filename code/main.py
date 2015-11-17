import scipty.io as sio
import numpy as np

# Global definitions
HIDDEN_LAYER_SIZE = 200
OUT_LAYER_SIZE = 10

def open_train_data():
	train_data_raw = sio.loadmat('../dataset/train.mat')
	train_images = train_data_raw['train_images']
	train_images_formatted = train_images.reshape((784, 60000)).T
	train_labels = train_data_raw['train_labels']

	return train_images_formatted, train_labels

