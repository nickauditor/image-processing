from PIL import Image
import torch
from torchvision import transforms
import onnxruntime as ort

onnx_path = "./onnx/model.onnx" 
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
session = ort.InferenceSession(onnx_path, providers=providers)

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_image_path = 'girl-2.png'

image = Image.open(input_image_path)
image = image.resize((1024, 1024))
input_tensor = transform_image(image).unsqueeze(0)

input_np = input_tensor.numpy()
outputs = session.run(
    output_names=None,  # Let ONNX infer output names
    input_feed={session.get_inputs()[0].name: input_np}
)

pred = torch.from_numpy(outputs[0][0][0]).sigmoid()  # Apply sigmoid if not included in model
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
mask.save("mask.png")
