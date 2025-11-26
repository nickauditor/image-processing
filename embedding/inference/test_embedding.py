from torchvision import transforms
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

image_list = os.listdir('images')

image_size = 224

transform_image = transforms.Compose([
    transforms.Resize(image_size, interpolation=Image.BICUBIC),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),  # Converts to [0,1] float tensor (equivalent to rescale)
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

onnx_path = "./onnx/model.onnx" 
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
session = ort.InferenceSession(onnx_path, providers=providers)

input_image_path = 'girl-2.png'
embedding_list = []

for img in image_list:
    input_image_path = 'images/' + img

    image = Image.open(input_image_path)

    input_tensor = transform_image(image).unsqueeze(0)
    input_np = input_tensor.numpy()

    outputs = session.run(
        output_names=None,  # Let ONNX infer output names
        input_feed={session.get_inputs()[0].name: input_np}
    )

    last_hidden_state = outputs[0]
    vecs = last_hidden_state[:, 0]  # shape (batch_size, embedding_dim)

    # Compute L2 norm along axis 1, keepdims=True for broadcasting
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)

    # print(norms)

    # Normalize vectors
    img_embeddings = vecs / norms
    embedding_list.append(img_embeddings[0])

np.save("embedding.npy", img_embeddings, allow_pickle=True)

test_image = 'images/images (4)_result.jpg'

image = Image.open(test_image)

input_tensor = transform_image(image).unsqueeze(0)
input_np = input_tensor.numpy()

outputs = session.run(
    output_names=None,  # Let ONNX infer output names
    input_feed={session.get_inputs()[0].name: input_np}
)

last_hidden_state = outputs[0]
vecs = last_hidden_state[:, 0]  # shape (batch_size, embedding_dim)

# Compute L2 norm along axis 1, keepdims=True for broadcasting
norms = np.linalg.norm(vecs, axis=1, keepdims=True)

# Normalize vectors
img_embeddings = vecs / norms
test_embedding = img_embeddings[0]

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = []

for embedding in embedding_list:
    similarity.append(cosine_similarity(test_embedding, embedding))

sorted_indices = np.argsort(-np.array(similarity))

for index in sorted_indices:
    print(image_list[index])
