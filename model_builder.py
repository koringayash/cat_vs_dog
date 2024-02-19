import torch
import torchvision
import torch.nn as nn
import torchvision.models as models

def set_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device

def importing_model():
  model_ft = models.vgg16(weights='VGG16_Weights.DEFAULT')
  for param in model_ft.parameters():
    param.requires_grad = False

  n_inputs = model_ft.classifier[6].in_features

  model_ft.classifier[6] = nn.Sequential(
  nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
  nn.Linear(256, 2), nn.Softmax(dim=1))

  model_ft = model_ft.to(set_device())

  return model_ft
