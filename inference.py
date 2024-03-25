import os

import numpy as np
from termcolor import colored
import cv2

import torch
import torchvision

from torch.nn import BCELoss, ReLU
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from sklearn.metrics import confusion_matrix, roc_auc_score

from data_loader import sampleBatches, sampleBatchesEvaluation
from models import getModel
from termcolor import colored

AUTHENTIC_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Au"
TAMPERED_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Tp"
GT_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_Groundtruth"

np.random.seed(12)

def main():

	# Setting the GPU
	gpu_ids = "0"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

	num_gpus = torch.cuda.device_count()
	print("Num GPU's:", num_gpus)

	gpu_indexes = np.arange(num_gpus).tolist()
	#gpu_indexes = np.array(list(map(int, gpu_ids.split(","))))
	print("Allocated GPU's for model:", gpu_indexes)

	val_images_paths = np.load("validation_paths.npy")

	# Defining data loader
	img_height = 128
	img_width = 128
	quality = 90
	
	breakpoint()
	val_sampler = sampleBatchesEvaluation(val_images_paths, img_height, img_width, quality)

	authentic_idxes = np.where(val_images_paths[:,1] == '0')[0]
	forgery_idxes = np.where(val_images_paths[:,1] == '1')[0]

	# Authentic images activations
	img_idx = np.random.choice(authentic_idxes)
	img_path = val_images_paths[img_idx][0]
	activations_visualization(val_sampler, img_path, img_idx, gpu_indexes)

	img_idx = np.random.choice(authentic_idxes)
	img_path = val_images_paths[img_idx][0]
	activations_visualization(val_sampler, img_path, img_idx, gpu_indexes)

	# Forgery images activations
	img_idx = np.random.choice(forgery_idxes)
	img_path = val_images_paths[img_idx][0]
	activations_visualization(val_sampler, img_path, img_idx, gpu_indexes)

	img_idx = np.random.choice(forgery_idxes)
	img_path = val_images_paths[img_idx][0]
	activations_visualization(val_sampler, img_path, img_idx, gpu_indexes)

def activations_visualization(val_sampler, img_path, img_idx, gpu_indexes):

	XAI = True
	img, label = val_sampler.__getitem__(img_idx)

	ela_image = val_sampler.convert_to_ela_image(img_path)
	ela_image_tensor = ToTensor()(ela_image)
	ela_image_tensor = np.uint8(ela_image_tensor*255)
	ela_image_opencv = np.moveaxis(ela_image_tensor, 0, -1)

	img = torch.stack([img]).cuda(gpu_indexes[0])
	
	# Loading model
	model = getModel(gpu_indexes, "resnet18")
	model.load_state_dict(torch.load("best_checkpoint_90.h5"))
	model.eval()


	prediction = model(img, XAI)

	print(colored("Probability of forgery: {:.2%}".format(prediction[0][0]), "yellow"))

	# get the gradient of the output with respect to the parameters of the model
	prediction[0].backward()

	# pull the gradients out of the model
	gradients = model.module.get_activations_gradient()

	# pool the gradients across the channels
	pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

	# get the activations of the last convolutional layer
	activations = model.module.get_activations(img).detach()

	# weight the channels by corresponding gradients
	for i in range(512):
		activations[:, i, :, :] *= pooled_gradients[i]
		
	# average the channels of the activations
	heatmap = torch.mean(activations, dim=1).squeeze()

	# relu on top of the heatmap
	# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
	heatmap = ReLU()(heatmap)

	# normalize the heatmap
	eps = 1e-9
	heatmap /= torch.max(heatmap)+eps

	# Converting the heatmap to numpy array
	heatmap = heatmap.cpu().detach().numpy()

	# Resizing and blending the activation map to the original image
	img = cv2.imread(img_path)
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
	heatmap = np.uint8(255 * heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	superimposed_img = heatmap * 0.4 + img

	if label == '1':
		# cv2.imread(img_path)
		gt_path = os.path.join(GT_DIR, img_path.split("/")[-1][:-4] + '_gt.png')
		gt = cv2.imread(gt_path)
		vis = np.concatenate((img, ela_image_opencv, superimposed_img, gt), axis=1)
		cv2.imwrite('map_%d.jpg' % img_idx, vis)
	else:
		vis = np.concatenate((img, superimposed_img, ela_image_opencv), axis=1)
		cv2.imwrite('map_%d.jpg' % img_idx, vis)



if __name__ == '__main__':
	main()
