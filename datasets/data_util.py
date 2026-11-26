import os
import numpy as np
import cv2
import torch

def exists(fs):
    for f in fs:
        if not os.path.exists(f):
            return False
    return True
def to_gpu(param, queue):
    data, key, device = param
    if isinstance(data[key], torch.Tensor):
        data[key] = data[key].to(device)
    queue.put({key: data[key]})
def get_circular_mask(h, w, percentage):
    # Desired masked area
    target_area = h * w * percentage

    # Area of circle = pi * r^2
    radius = int(np.sqrt(target_area / np.pi))
    x = np.random.randint(0, w - 1)
    y = np.random.randint(0, h - 1)
    # Binary mask
    mask = np.ones((h, w), dtype=np.uint8)

    cv2.circle(
        mask,
        (x, y),
        radius,
        0,
        thickness=-1
    )
    return mask

def image_to_points(image):
    """
    Convert grayscale image to:
        [x, y, value]
    """

    # Convert to grayscale if necessary
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape

    # Generate coordinates
    y, x = np.indices((h, w))

    # Create [x, y, value]
    points = np.column_stack((
        x.ravel(),
        y.ravel(),
        image.ravel()
    ))

    return points

def multiply_matching_coordinates_rgb(array1, array2):
    # Map (x, y) -> value from array2
    value_map = {
        (x, y): v
        for x, y, v in array2
    }

    result = []

    for x, y, r,g,b in array1:
        if (x, y) in value_map:
            v2 = value_map[(x, y)]

            result.append([
                r * v2, g * v2, b * v2
            ])
        else:
            result.append([
                r , g , b
            ])

    return np.array(result)

def multiply_matching_coordinates_thermal(array1, array2):
    # Map (x, y) -> value from array2
    value_map = {
        (x, y): v
        for x, y, v in array2
    }

    result = []

    for x, y, t in array1:
        if (x, y) in value_map:
            v2 = value_map[(x, y)]

            result.append([
                t * v2
            ])
        else:
            result.append(t)

    return np.array(result).reshape(-1, 1)

def get_circular_mask_center(h, w, percentage):
    # Desired masked area
    target_area = h * w * percentage

    # Area of circle = pi * r^2
    radius = int(np.sqrt(target_area / np.pi))
    x = int(w / 2)
    y = int(h / 2)
    # Binary mask
    mask = np.ones((h, w), dtype=np.uint8)

    cv2.circle(
        mask,
        (x, y),
        radius,
        0,
        thickness=-1
    )
    return mask

def to_gpu(param, queue):
    data, key, device = param
    if isinstance(data[key], torch.Tensor):
        data[key] = data[key].to(device)
    queue.put({key: data[key]})