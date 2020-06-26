import torchvision
import numpy as np
from PIL import Image

def image_to_tensor(img_path, max_size=200, shape=None):
    
  image = Image.open(img_path).convert('RGB')
  
  if max(image.size) > max_size:
    size = max_size

  else:
    size = max(image.size)
	
  if shape is not None:
    size = shape
  
  transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((size, int(1.5*size))),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
  
  return transform(image)[:3, :, :].unsqueeze(0)

def tensor_to_image(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.numpy().squeeze()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

  return image.clip(0, 1)

def get_features(image, model, layers=None):
    """
    Content Reconstructions: use conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1
    Style Reconstructions: use different subsets of conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1
    """
    
    pass

model = torchvision.models.vgg19(pretrained=True, progress=True)

get_features(image_to_tensor('Hokusai.jpg'), model)


