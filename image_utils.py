from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    
    image = Image.open(path)   
    image_array = np.array(image)    
    return image_array


def edge_detection(image_array):
   
    gray = image_array.astype(np.float32).mean(axis=2)

    kernelY = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32)

    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    edgeX = convolve2D(gray, kernelX, mode='constant', cval=0.0)
    edgeY = convolve2D(gray, kernelY, mode='constant', cval=0.0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
