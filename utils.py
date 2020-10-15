import cv2


import torch
import numpy as np

from torchvision import transforms


def generate_trimap(probs, size, conf_threshold):
    # trimap = (probs > 0.05).astype(float) * 0.5

    pixels = 2 * size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels, pixels))

    fg = (probs > conf_threshold).astype(float)
    unknown = np.logical_and(probs > 0.1, probs <= conf_threshold).astype(float) * 0.5

    cv2.erode(fg, kernel, iterations=5)
    cv2.dilate(unknown, kernel, iterations=5)

    trimap = fg + unknown
    trimap = cv2.morphologyEx(trimap, cv2.MORPH_CLOSE, kernel, iterations=5)

    # trimap = cv2.dilate(trimap, kernel, iterations=1)
    # trimap = cv2.morphologyEx(trimap, cv2.MORPH_CLOSE, kernel, iterations=1)
    # trimap[probs > conf_threshold] = 1

    return trimap


def image_to_trimap(image, model, return_seg_mask=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    batch: torch.Tensor = preprocess(image).unsqueeze(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch = batch.to(device)
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        output: torch.Tensor = model(batch)["out"][0].softmax(dim=0)

    fg_probs = (1.0 - output[0]).cpu().numpy()

    if return_seg_mask:
        predictions = output.argmax(0).cpu().numpy()
        seg_mask = np.vectorize(lambda pix: 0 if pix == 0 else 255)(predictions)[
            ..., np.newaxis
        ]
        return generate_trimap(fg_probs, 5, 0.925), seg_mask

    return generate_trimap(fg_probs, 5, 0.925)


# def extract_alpha(image,trimap):


def fba_trimap(trimap):
    h, w = trimap.shape
    fba_trimap = np.zeros((h, w, 2))
    fba_trimap[trimap == 1, 1] = 1
    fba_trimap[trimap == 0, 0] = 1
    return fba_trimap



