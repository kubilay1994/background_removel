import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2
from pymatting.alpha import estimate_alpha_cf, estimate_alpha_knn


from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101


from utils import generate_trimap

model: torch.nn.Module = deeplabv3_resnet101(pretrained=True)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


img = Image.open("images/cat.jpg")
img_array = np.array(img) / 255

input_tensor: torch.Tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input_batch = input_batch.to(device)
model = model.to(device)

with torch.no_grad():
    model.eval()
    output: torch.Tensor = model(input_batch)["out"][0].softmax(dim=0)


predictions = output.argmax(0)

fg_probs = (1.0 - output[0]).cpu().numpy()


trimap = generate_trimap(fg_probs, 1, 0.95)


alpha = estimate_alpha_knn(
    img_array,
    trimap,
    # laplacian_kwargs={"epsilon": 1e-6},
    # cg_kwargs={"maxiter":2000}
)


cv2.imshow("trimap", cv2.resize(trimap, (600, 600)))
cv2.imshow("alpha", cv2.resize(alpha, (600, 600)))
cv2.waitKey(0)
# arr = predictions.cpu().numpy()