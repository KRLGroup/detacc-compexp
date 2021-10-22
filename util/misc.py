"""
misc utils

Reference: https://github.com/jayelm/compexp/blob/master/vision/util/misc.py

New functions: upsample, get_unit_acts
"""
import numpy as np
from PIL import Image


def safe_layername(layer):
    if isinstance(layer, list):
        return "-".join(map(str, layer))
    else:
        return layer


def upsample(features, shape):
    """ 
    Resizes 'features' to 'shape'
    Args:
        features: numpy array to be reshaped
        shape: tuple of new shape
    Returns:
        a new numpy array of features with the new shape
    """
    return np.array(Image.fromarray(features).resize(shape, resample=Image.BILINEAR))


def get_unit_acts(ufeat, uthresh, mask_shape, data_size):
    """ 
    Computes unit activations from features and thresholds
    Args:
        ufeat: (ndarray) features of a single unit over the dataset
        uthresh: (float) the activation threshold for this unit
        mask_shape: (tuple) new dimension to upsample each image to
        data_size: (int) number of images in dataset

    """
    uidx = np.argwhere(ufeat.max((1, 2)) > uthresh).squeeze(1)
    ufeat = np.array([upsample(ufeat[i], mask_shape) for i in uidx])

    # Create full array
    uhitidx = np.zeros((data_size, *mask_shape), dtype=np.bool)

    # Change mask to bool based on threshold
    uhit_subset = ufeat > uthresh    
    uhitidx[uidx] = uhit_subset

    return uhitidx

