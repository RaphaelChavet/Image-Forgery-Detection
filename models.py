from torchvision.models import resnet18, resnet152
from torchvision.models import ResNet18_Weights, ResNet152_Weights
from torch.nn import Module, Sigmoid, Linear, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torch.nn import DataParallel
from spliceburster.src import SB_launcher, SB_out2uint8, SB_showout, SB 

def getModel(gpu_indexes, model_name):
	
	if model_name == 'resnet18':
		model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		model = ResNet18(model).cuda()
	elif model_name == 'resnet152':
		model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
		model = ResNet152(model).cuda()
	elif model_name =="splicebuster":
		#launch SB_launcher and exit this programm...
		exit

	model.eval()
	model = DataParallel(model, device_ids=gpu_indexes)
	return model

class ResNet18(Module):
    
	def __init__(self, model_base):
		super(ResNet18, self).__init__()

		self.conv1 = model_base.conv1
		self.bn1 = model_base.bn1
		self.relu = model_base.relu
		self.maxpool = model_base.maxpool
		self.layer1 = model_base.layer1
		self.layer2 = model_base.layer2
		self.layer3 = model_base.layer3
		self.layer4 = model_base.layer4
		self.avgpool = model_base.avgpool

		self.classification_layer = Linear(512, 2, bias=True)

		# placeholder for the gradients
		self.gradients = None
		
	# hook for the gradients of the activations
	def activations_hook(self, grad):
		self.gradients = grad

	def features_conv(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x) 
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		return x
		
	def forward(self, x, XAI=False):
		
		x = self.features_conv(x)

		# register the hook
		if XAI:
			h = x.register_hook(self.activations_hook)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		
		#prediction = Sigmoid()(self.classification_layer(x))
		prediction = self.classification_layer(x)
		
		return prediction
	
	# method for the gradient extraction
	def get_activations_gradient(self):
		return self.gradients
    
    # method for the activation exctraction
	def get_activations(self, x):
		return self.features_conv(x)


class ResNet152(Module):
    
	def __init__(self, model_base):
		super(ResNet152, self).__init__()


		self.conv1 = model_base.conv1
		self.bn1 = model_base.bn1
		self.relu = model_base.relu
		self.maxpool = model_base.maxpool
		self.layer1 = model_base.layer1
		self.layer2 = model_base.layer2
		self.layer3 = model_base.layer3
		self.layer4 = model_base.layer4

		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))

		self.classification_layer = Linear(2048, 2, bias=True)

	
	def forward(self, x):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		#x = self.spatial_channel_attention(x, "layer4")
		
		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg #+ x_max
		
		x = x.view(x.size(0), -1)
		
		prediction = self.classification_layer(x)
		return prediction

	def spatial_channel_attention(self, x, layer_name):

		if layer_name == "layer4":
			space_attention = torch.sigmoid(self.conv1x1_layer4(x))

			x_avg = self.global_avgpool(x)
			x_max = self.global_maxpool(x)
			gp = torch.cat((x_avg, x_max), dim=1)


			#NSA
			squeezed_fv = torch.nn.ReLU(inplace=True)(self.conv1x1_squeeze_layer4(gp))
			channel_attention = torch.sigmoid(self.conv1x1_expand_layer4(squeezed_fv))

		attention_x = x*space_attention + x*channel_attention + x

		return attention_x