import cv2
import random


def generate_trimap(probs, size, conf_threshold):
    trimap = (probs > 0.05).astype(float) * 0.5

    pixels = 2 * size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels, pixels))


    # iterations = random.randint(1, 10)
    trimap = cv2.morphologyEx(trimap, cv2.MORPH_CLOSE, kernel, iterations=1)
    trimap[probs > conf_threshold] = 1

    return trimap
