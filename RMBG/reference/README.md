---
license: other
license_name: bria-rmbg-2.0
pipeline_tag: image-segmentation
tags:
- remove background
- background
- background-removal
- Pytorch
- vision
- legal liability
- transformers
- transformers.js
extra_gated_description: >-
  Bria AI Model weights are open source for non commercial use only, per the
  provided license.
extra_gated_heading: Fill in this form to immediatly access the model for non commercial use
extra_gated_fields:
  Name: text
  Email: text
  Company/Org name: text
  Company Website URL: text
  Discord user: text
  I agree to BRIA’s Privacy policy, Terms & conditions, and acknowledge Non commercial use to be Personal use / Academy / Non profit (direct or indirect): checkbox
---

# BRIA Background Removal v2.0 Model Card

RMBG v2.0 is our new state-of-the-art background removal model significantly improves RMBG v1.4. The model is designed to effectively separate foreground from background in a range of
categories and image types. This model has been trained on a carefully selected dataset, which includes:
general stock images, e-commerce, gaming, and advertising content, making it suitable for commercial use cases powering enterprise content creation at scale. 
The accuracy, efficiency, and versatility currently rival leading source-available models. 
It is ideal where content safety, legally licensed datasets, and bias mitigation are paramount. 

Developed by BRIA AI, RMBG v2.0 is available as a source-available model for non-commercial use.

### Get Access

Bria RMBG2.0 is availabe everywhere you build, either as source-code and weights, ComfyUI nodes or API endpoints.

## Model Details
#####
### Model Description

- **Developed by:** BRIA AI
- **Model type:** Background Removal 
- **License:** Creative Commons Attribution–Non-Commercial (CC BY-NC 4.0)
  - The model is released under a CC BY-NC 4.0 license for non-commercial use.
  - Commercial use is subject to a commercial agreement with BRIA. Available here.

- **Model Description:** BRIA RMBG-2.0 is a dichotomous image segmentation model trained exclusively on a professional-grade dataset. The model output includes a single-channel 8-bit grayscale alpha matte, where each pixel value indicates the opacity level of the corresponding pixel in the original image. This non-binary output approach offers developers the flexibility to define custom thresholds for foreground-background separation, catering to varied use cases requirements and enhancing integration into complex pipelines.
- **BRIA:** Resources for more information: BRIA AI.


## Training data
Bria-RMBG model was trained with over 15,000 high-quality, high-resolution, manually labeled (pixel-wise accuracy), fully licensed images.
Our benchmark included balanced gender, balanced ethnicity, and people with different types of disabilities.
For clarity, we provide our data distribution according to different categories, demonstrating our model’s versatility.

### Distribution of images:

| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Objects only | 45.11% |
| People with objects/animals | 25.24% |
| People only | 17.35% |
| people/objects/animals with text | 8.52% |
| Text only | 2.52% |
| Animals only | 1.89% |

| Category | Distribution |
| -----------------------------------| -----------------------------------------:|
| Photorealistic | 87.70% |
| Non-Photorealistic | 12.30% |


| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Non Solid Background | 52.05% |
| Solid Background | 47.95% 


| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Single main foreground object | 51.42% |
| Multiple objects in the foreground | 48.58% |


## Qualitative Evaluation
Open source models comparison
![diagram](diagram1.png)

### Architecture
RMBG-2.0 is developed on the **BiRefNet** architecture enhanced with our proprietary dataset and training scheme. This training data significantly improves the model’s accuracy and effectiveness for background-removal task.<br>


#### Requirements
```bash
torch
torchvision
pillow
kornia
transformers
```

### Usage

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->


```python
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True).eval().to(device)

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(input_image_path)
input_images = transform_image(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid().cpu()
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
image.putalpha(mask)

image.save("no_bg_image.png")
```


</div>
