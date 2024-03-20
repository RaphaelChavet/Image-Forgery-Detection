import os

import numpy as np
from termcolor import colored

import torch
import torchvision

from torch.nn import BCELoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from sklearn.metrics import confusion_matrix, roc_auc_score

from data_loader import sampleBatches, sampleBatchesEvaluation, dataset_processing
from models import getModel

import cv2

AUTHENTIC_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Au"
TAMPERED_DIR = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Tp"

np.random.seed(12)

def main():

	# Setting the GPU
	gpu_ids = "6,7"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

	num_gpus = torch.cuda.device_count()
	print("Num GPU's:", num_gpus)

	gpu_indexes = np.arange(num_gpus).tolist()
	#gpu_indexes = np.array(list(map(int, gpu_ids.split(","))))
	print("Allocated GPU's for model:", gpu_indexes)

	img_height = 512
	img_width = 512
	quality = 90

	data_manager = dataset_processing()
	train_images_paths, val_images_paths, test_images_paths = data_manager.get_sets()

	# Defining data loader
	img_height = 128
	img_width = 128
	quality = 90
	
	batch_size = 1024
	train_sampler = sampleBatches(train_images_paths, img_height, img_width, quality)
	trainLoader = DataLoader(train_sampler, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True, collate_fn=collate_samples)
	num_iterations = len(trainLoader)

	val_sampler = sampleBatchesEvaluation(val_images_paths, img_height, img_width, quality)
	valLoader = DataLoader(val_sampler, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False, drop_last=False, collate_fn=collate_samples)

	# Some visualizations of images and their respective ELA images
	'''
	img_path = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Au/Au_arc_30714.jpg"
	save_original_and_ela_images(img_path, val_sampler)

	img_path = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Au/Au_nat_30090.jpg"
	save_original_and_ela_images(img_path, val_sampler)

	img_path = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Tp/Tp_D_NRN_M_N_arc00010_nat00062_11176.jpg"
	save_original_and_ela_images(img_path, val_sampler)
	plot_forgery_formation(img_path)

	img_path = "/hadatasets/gabriel_bertocco/ForensicsDatasets/CASIA2.0/CASIA2.0_revised/Tp/Tp_D_NRN_M_N_nat10116_cha00031_11368.jpg"
	save_original_and_ela_images(img_path, val_sampler)
	plot_forgery_formation(img_path)
	'''

	# Loading model
	#model = getModel(gpu_indexes, "resnet18")
	model = getModel(gpu_indexes, "resnet152")
	model.train()

	#bce_loss = BCELoss(reduction='mean')
	ce_loss = CrossEntropyLoss(reduction='mean')
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

	num_epochs = 300
	best_acc_bal = 0.0

	for epoch_counter in range(num_epochs):

		batch_counter = 0
		for batch_images, batch_labels in trainLoader:
			batch_images = batch_images.cuda(gpu_indexes[0])
			# Reshape to match dimensions to the predictions (preds)
			#batch_labels = batch_labels.reshape(batch_labels.shape[0], 1).cuda(gpu_indexes[0])
			batch_labels = batch_labels.long().cuda(gpu_indexes[0])

			preds = model(batch_images)
			#batch_loss = bce_loss(preds, batch_labels)
			batch_loss = ce_loss(preds, batch_labels)

			# Zero all gradients
			optimizer.zero_grad()
			# Calculate and backward gradients
			batch_loss.backward()
			# Perform the weights updating
			optimizer.step()

			batch_counter += 1

			TNR, TPR, FPR, FNR, acc_bal = calculate_balanced_accuracy_and_auc(preds, batch_labels)
			print("Epoch: {}/{}, Iteration: {}/{}, Batch Loss: {:.7}, TPR: {:.2%}, TNR: {:.2%}, ACC_BAL: {:.2%}".format(epoch_counter+1, num_epochs, 
																																	batch_counter, num_iterations, 
																																	batch_loss.item(), TPR, TNR, 
																																	acc_bal))
		
		print("Evaluting in validation set ...")
		# Evaluation on validation set
		model.eval()
		validation_predictions = []
		validation_labels = []
		for batch_images, batch_labels in valLoader:
			batch_images = batch_images.cuda(gpu_indexes[0])
			# Reshape to match dimensions to the predictions (preds)
			#batch_labels = batch_labels.reshape(batch_labels.shape[0], 1).cuda(gpu_indexes[0])
			batch_labels = batch_labels.long().cuda(gpu_indexes[0])

			with torch.no_grad():
				preds = model(batch_images)

			validation_predictions.append(preds)
			validation_labels.append(batch_labels)

		
		validation_predictions = torch.cat(validation_predictions, dim=0)
		validation_labels = torch.cat(validation_labels, dim=0)

		TNR, TPR, FPR, FNR, acc_bal = calculate_balanced_accuracy_and_auc(validation_predictions, validation_labels)
		print(colored("Epoch: {}/{}, TPR: {:.2%}, TNR: {:.2%}, ACC_BAL: {:.2%}".format(epoch_counter+1, num_epochs, TPR, TNR, acc_bal), 'yellow'))

		if acc_bal > best_acc_bal:
			best_acc_bal = acc_bal
			best_checkpoint_epoch = epoch_counter
			torch.save(model.state_dict(), "best_checkpoint.h5")
		
		print("Best checkpoint so far is from epoch {}".format(best_checkpoint_epoch+1))

		model.train()


	# Evaluation on test set
	print("Evaluating best checkpoint in test set ...")

	# Loading best checkpoint
	model.load_state_dict(torch.load("best_checkpoint.h5"))

	test_sampler = sampleBatchesEvaluation(test_images_paths, img_height, img_width, quality)
	testLoader = DataLoader(test_sampler, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False, drop_last=False, collate_fn=collate_samples)

	model.eval()
	test_predictions = []
	test_labels = []
	for batch_images, batch_labels in testLoader:
		batch_images = batch_images.cuda(gpu_indexes[0])
		# Reshape to match dimensions to the predictions (preds)
		#batch_labels = batch_labels.reshape(batch_labels.shape[0], 1).cuda(gpu_indexes[0])
		batch_labels = batch_labels.long().cuda(gpu_indexes[0])

		with torch.no_grad():
			preds = model(batch_images)

		test_predictions.append(preds)
		test_labels.append(batch_labels)

	
	test_predictions = torch.cat(test_predictions, dim=0)
	test_labels = torch.cat(test_labels, dim=0)

	TNR, TPR, FPR, FNR, acc_bal = calculate_balanced_accuracy_and_auc(test_predictions, test_labels)
	print(colored("Epoch: {}/{}, TPR: {:.2%}, TNR: {:.2%}, ACC_BAL: {:.2%}".format(epoch_counter, num_epochs, TPR, TNR, acc_bal), 'cyan'))


def save_original_and_ela_images(img_path, val_sampler):

	ela_image = val_sampler.convert_to_ela_image(img_path)
	ela_image_tensor = ToTensor()(ela_image)
	ela_image_tensor = np.uint8(ela_image_tensor*255)
	ela_image_opencv = np.moveaxis(ela_image_tensor, 0, -1)

	img = cv2.imread(img_path)

	output_name = img_path.split("/")[-1][:-4]
		
	vis = np.concatenate((img, ela_image_opencv), axis=1)
	cv2.imwrite('original_and_ela_%s.jpg' % output_name, vis)
	
def plot_forgery_formation(img_forgery_path):

	img_path_parts = img_forgery_path.split("/")[-1].split("_")
	img_name01 = img_path_parts[5]
	img_name02 = img_path_parts[6]

	img_name01 = "Au_" + img_name01[:3] + "_" + img_name01[3:] + ".jpg"
	img_name02 = "Au_" + img_name02[:3] + "_" + img_name02[3:] + ".jpg"

	full_img01_path = os.path.join(AUTHENTIC_DIR, img_name01)
	full_img02_path = os.path.join(AUTHENTIC_DIR, img_name02)

	print("Source image path: %s" % full_img01_path)
	print("Target image path: %s" % full_img02_path)




def calculate_balanced_accuracy_and_auc(preds, labels, threshold=0.5):

	#y_pred = (preds >= threshold).cpu().numpy().flatten().astype(int)
	#preds_cpu = preds.detach().cpu().numpy().flatten()
	#y_true = labels.cpu().numpy().flatten()

	y_pred = preds.argmax(dim=1).cpu().numpy()
	y_true = labels.cpu().numpy()

	# Confusion Matrix
	CM = confusion_matrix(y_true, y_pred, normalize='true')

	# Metrics
	TNR = CM[0][0]
	TPR = CM[1][1]
	FPR = CM[0][1]
	FNR = CM[1][0]
	acc_bal = (TNR + TPR)/2

	#AUC = roc_auc_score(y_true, preds_cpu)
	
	return TNR, TPR, FPR, FNR, acc_bal #, AUC

def collate_samples(batch):
	
	images = []
	labels = []

	for img, label in batch:
		images.append(torch.stack([img]))
		labels.append(int(label))

	images = torch.cat(images, dim=0)
	labels = torch.Tensor(labels)

	return images, labels



if __name__ == '__main__':
	main()
