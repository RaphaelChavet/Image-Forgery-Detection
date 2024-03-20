import os
import random
from torch.utils.data import Dataset, DataLoader

from io import BytesIO
from PIL import Image, ImageChops, ImageEnhance 
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, RandomErasing, RandomHorizontalFlip, ColorJitter
from torchvision.transforms import RandomCrop, functional, GaussianBlur, Grayscale, ToPILImage, RandomGrayscale, GaussianBlur

import numpy as np

AUTHENTIC_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Au"
TAMPERED_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Tp"

ROOT_IMD2020 = "/hadatasets/gabriel_bertocco/ForensicsDatasets/ImageForgery/IMD2020/large_scale_forgery_detection"
ROOT_CASIA02 = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised"


# Define a custom transform for blurring with 50% probability
class RandomBlur:

	def __init__(self, probability=0.5):
		self.probability = probability
		self.gaussian_blur = GaussianBlur((5,5), sigma=1.0)

	def __call__(self, img):
		if random.random() >= self.probability:
			return self.gaussian_blur(img) 
		return img

class sampleBatches(Dataset):
    
	def __init__(self, images_paths, img_height, img_width, quality):
		self.images_paths = images_paths
		self.img_height = img_height
		self.img_width = img_width
		self.quality = quality

		np.random.shuffle(self.images_paths)
		
		self.transform = Compose([Resize((img_height, img_width), interpolation=functional.InterpolationMode.BICUBIC), 
									#RandomCrop((img_height, img_width), padding=10), 
									RandomHorizontalFlip(p=0.5),
									RandomBlur(probability=0.5), 
									ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.0), 
									RandomGrayscale(p=0.5),
									ToTensor(),
									#RandomErasing(p=1.0, scale=(0.02, 0.33)),
									Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	def __getitem__(self, idx):

		img_path = self.images_paths[idx][0]
		img_label = self.images_paths[idx][1]
		imgPIL = self.convert_to_ela_image(img_path)
		img = self.transform(imgPIL)
		return img, img_label


	def convert_to_ela_image(self, path):

		original_image = Image.open(path).convert('RGB')

		buffer = BytesIO()

		#resaved_file_name = os.path.join('temp_images', path.split("/")[-1][:-4] + 'resaved_image.jpg')   
		#original_image.save(resaved_file_name,'JPEG',quality=self.quality)
		original_image.save(buffer,'JPEG',quality=self.quality)

		# Rewind the buffer to the beginning
		buffer.seek(0)
		resaved_image = Image.open(buffer)
		#resaved_image = Image.open(resaved_file_name)
		#os.remove(resaved_file_name)

		ela_image = ImageChops.difference(original_image,resaved_image)
		
		extrema = ela_image.getextrema()
		max_difference = max([pix[1] for pix in extrema])
		if max_difference ==0:
			max_difference = 1
		scale = 255 / max_difference
		
		ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

		return ela_image
		
	def __len__(self):
		return len(self.images_paths)
	

class sampleBatchesEvaluation(sampleBatches):

	def __init__(self, images_paths, img_height, img_width, quality):
		super(sampleBatchesEvaluation, self).__init__(images_paths, img_height, img_width, quality)

		self.transform = Compose([Resize((img_height, img_width), interpolation=functional.InterpolationMode.BICUBIC), 
										ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class dataset_processing:
    
	def __init__(self):
		
		images_dict = {'Authentic': {}, 'Forgery': {}}

		##### Processing IMD2020 #######
		for fd in os.listdir(ROOT_IMD2020):
			fd_path = os.path.join(ROOT_IMD2020, fd)
			if os.path.isdir(fd_path):
				images_dict_partial = self.check_dir_get_images(fd_path)

				if 'Generative' in fd:
					key = 'Forgery'
				else:
					key = 'Authentic'

				images_dict[key][fd] = images_dict_partial

	
		##### Processing CASIA2.0 #######
		for fd in os.listdir(ROOT_CASIA02):
			fd_path = os.path.join(ROOT_CASIA02, fd)
			if os.path.isdir(fd_path):
				images_dict_partial = self.check_dir_get_images(fd_path)

				if fd == 'Tp':
					key = 'Forgery'
				else:
					key = 'Authentic'

				images_dict[key][fd] = images_dict_partial

		self.images_dict = images_dict
		self.authentic_images = self.get_array_all_images(self.images_dict['Authentic'])
		self.forgery_images = self.get_array_all_images(self.images_dict['Forgery'])

		print("There are {} authentic images and {} forged images".format(self.authentic_images.shape[0], self.forgery_images.shape[0]))

		train_authentic_images, val_authentic_images, test_authentic_images = self.train_val_test_division(self.authentic_images, label=0)
		train_forgery_images, val_forgery_images, test_forgery_images = self.train_val_test_division(self.forgery_images, label=1)

		print("there are {} authentic images and {} forged images in the training set".format(train_authentic_images.shape[0], train_forgery_images.shape[0]))
		print("there are {} authentic images and {} forged images in the validation set".format(val_authentic_images.shape[0], val_forgery_images.shape[0]))
		print("there are {} authentic images and {} forged images in the test set".format(test_authentic_images.shape[0], test_forgery_images.shape[0]))

		self.training_set = np.concatenate((train_authentic_images, train_forgery_images), axis=0)
		self.validation_set = np.concatenate((val_authentic_images, val_forgery_images), axis=0)
		self.test_set = np.concatenate((test_authentic_images, test_forgery_images), axis=0)

		np.random.shuffle(self.training_set)
		
	def check_dir_get_images(self, dir_level):

		if not os.path.isdir(dir_level):
			return dir_level

		intermediate_dict = {}
		level_files_and_dirs = os.listdir(dir_level)

		for lfd in level_files_and_dirs:
			next_level = os.path.join(dir_level, lfd)
			content_next_level = self.check_dir_get_images(next_level)

			if type(content_next_level) == dict:
				intermediate_dict[lfd] = content_next_level
			else:
				if content_next_level.endswith('.jpg') or content_next_level.endswith('.png'):
					intermediate_dict[content_next_level[:-4]] = content_next_level
		
		return intermediate_dict

	
	def get_array_all_images(self, images_subdict):

		if not type(images_subdict) == dict:
			if images_subdict.endswith('.jpg') or images_subdict.endswith('.png'):
				return np.array([images_subdict])

		all_images_paths = []
		for key in images_subdict.keys():
			images_paths = self.get_array_all_images(images_subdict[key])
			all_images_paths.append(images_paths)
		
		all_images_paths = np.concatenate(all_images_paths, axis=0)
		return all_images_paths

	def train_val_test_division(self, images_paths, label=0):


		N = images_paths.shape[0]
		images_paths = np.array([[img_path, label] for img_path in images_paths])
		print("Total number of images: %d" % N)

		# Train/Val/Test split
		samples_idx = np.arange(N)
		np.random.shuffle(samples_idx)

		train_split_idx = int(N*0.7)
		val_split_idx = train_split_idx + int(N*0.1)

		train_idx = samples_idx[:train_split_idx]
		val_idx = samples_idx[train_split_idx:val_split_idx]
		test_idx = samples_idx[val_split_idx:]

		# Defining training set
		train_images_paths = images_paths[train_idx]
		
		# Validation set
		val_images_paths = images_paths[val_idx]
		
		# Test set
		test_images_paths = images_paths[test_idx]
		
		return train_images_paths, val_images_paths, test_images_paths

	def get_sets(self):
		return self.training_set, self.validation_set, self.test_set

	def save_sets(self):
		# Saving images paths
		np.save("training_paths.npy", train_images_paths)
		np.save("validation_paths.npy", val_images_paths)
		np.save("test_paths.npy", test_images_paths)
	
	def load_sets(self):
		self.training_set = np.load("training_paths.npy")
		self.validation_set = np.load("validation_paths.npy")
		self.test_set = np.load("test_paths.npy")


