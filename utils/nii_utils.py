"""
Functions for nii manipulation
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from nipy import load_image, save_image
from nipy.core.api import Image
import nibabel

def mask_nii_data(data, mask):
    """
    mask nii data from 3-D into 1-D
    """
    mask = mask.flatten()
    data = data.reshape(mask.shape[0], -1)
    data = data[mask > 0]
    return data

def unmask_nii_data(data, mask):
    """
    unmask nii data from 1-D into 3-D
    """
    shape = mask.shape
    mask = mask.flatten()

    # Format the new data
    value = np.zeros(mask.shape)
    value[mask > 0] = data

    return value.reshape(shape)

def load_nii_image(filename, mask=None):
    """
    Get data from nii image.
    """
    nii = load_image(filename)
    data = nii.get_data()

    if mask is not None:
        data = mask_nii_data(data, mask)

    return data

def save_nii_image(filename, data, template, mask=None):
    """
    Save data into nii image.
    """
    if mask is not None:
        data = unmask_nii_data(data, mask)

    tmp =load_image(template)
    img = Image(data, tmp.coordmap, tmp.metadata)
    try:
        save_image(img, filename)
    except IOError:
        path = '/'.join(filename.split('/')[:-1])
        os.system('mkdir -p ' + path)
        save_image(img, filename)

