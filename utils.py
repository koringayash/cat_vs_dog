import import_files
import torch
import torchvision
from torchvision import transforms

def set_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device

def func_load_model(model,path="vgg16_model.pth"):
  model.load_state_dict(torch.load(path, map_location='cpu'))
  return model


def custom_image(model,lst,path='geeks.jpg'):
  with torch.inference_mode():
    custom_image = torchvision.io.read_image(str(path)).type(torch.float32)
    custom_image = custom_image / 255.
    custom_image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])
    device = set_device()
    custom_image_transformed = custom_image_transform(custom_image)
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(device))
    print(custom_image_pred)
    return lst[torch.argmax(custom_image_pred)]
