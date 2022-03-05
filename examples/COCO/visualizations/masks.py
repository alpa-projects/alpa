import numpy as np
from PIL import ImageColor
import colorsys
import random
import cv2


def get_colors(N, shuffle=False, bright=True):
    """
    https://github.com/pedropro/TACO/blob/master/detector/visualize.py

    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    colors = [[0, 0, 0]]
    if N == 1:
        return colors

    brightness = 1.0 if bright else 0.7
    hsv = [(i / (N - 1), 1, brightness) for i in range(N - 1)]
    colors.extend(list(map(lambda c: list(map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*c))), hsv)))

    if shuffle:
        random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def overlay_classes(image, target, colors, num_classes, alpha=0.5):
    assert(len(colors) == num_classes)
    for k, color in enumerate(colors):
        mask = np.where(target == k, 1, 0)
        image = apply_mask(image, mask, color, alpha=alpha)
    return image


def get_colored_segmentation_mask(predictions, num_classes, images=None, binary_threshold=0.5, alpha=0.5):
    """
    Arguments:

    predictions: ndarray - BHWC (if C=1 - np.float32 else np.uint8) probabilities
    num_classes: int - number of classes
    images: ndarray - float32 (0..1) or uint8 (0..255)
    binary_threshold: float - when predicting only 1 value, threshold to set label to 1
    alpha: float - overlay percentage
    """
    predictions = predictions.copy()
    colors = get_colors(num_classes)

    if images is None:
        shape = (predictions.shape[0], predictions.shape[1], predictions.shape[2], 3)
        images = np.zeros(shape, np.uint8)
    else:
        images = images.copy()

        if images.dtype == "float32" or images.dtype == 'float64':
            images = (images * 255).astype(np.uint8)

        if images.shape[-1] == 1:
            images = [cv2.cvtColor(i.copy(), cv2.COLOR_GRAY2RGB) for i in images]
            images = np.asarray(images)

    if predictions.shape[-1] == 1:
        # remove channel dimension
        predictions = np.squeeze(predictions, axis=-1)

        # set either zero or one
        predictions[predictions > binary_threshold] = 1.0
        predictions[predictions <= binary_threshold] = 0.0
    else:
        # find the argmax channel from all channels
        predictions = np.argmax(predictions, axis=-1)

    predictions = predictions.astype(np.uint8)

    for i in range(len(predictions)):
        images[i, :, :, :] = overlay_classes(images[i, :, :, :].copy(), predictions[i], colors, num_classes, alpha=alpha)

    return images
