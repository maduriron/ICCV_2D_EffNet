import scipy.ndimage as ndimage
import scipy
import skimage.measure
import numpy as np
import os
import sys
import SimpleITK as sitk
# import pydicom as pyd
import logging
from tqdm import tqdm
import skimage.morphology
from PIL import Image


def simple_bodymask(img, maskthreshold=150):
    maskthreshold = maskthreshold
    oshape = img.shape
    img = ndimage.zoom(img, 0.25, order=0)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(int)
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.divide(oshape, img.shape)[0]
    return ndimage.zoom(bodymask, real_scaling, order=0)


def crop_and_resize(img, mask=None, width=192, height=192, maskthreshold=150):
    bmask = simple_bodymask(img, maskthreshold)
    # img[bmask==0] = -1024 # this line removes background outside of the lung.
    # However, it has been shown problematic with narrow circular field of views that touch the lung.
    # Possibly doing more harm than help
    reg = skimage.measure.regionprops(skimage.measure.label(bmask))
    if len(reg) > 0:
        bbox = reg[0].bbox
    else:
        bbox = (0, 0, bmask.shape[0], bmask.shape[1])
    img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    img = ndimage.zoom(img, np.asarray([width, height]) / np.asarray(img.shape), order=1)
    if not mask is None:
        mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask = ndimage.zoom(mask, np.asarray([width, height]) / np.asarray(mask.shape), order=0)
        # mask = ndimage.binary_closing(mask,iterations=5)
    return img, mask, bbox


class SmartCrop(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(256, 256), maskthreshold=150, np_array=True):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.maskthreshold = maskthreshold
        self.np_array = np_array
        self.i = 0

    def __call__(self, image):
        image = np.asarray(image)[:, :, 0]

        (image, _, _) = crop_and_resize(
            image, width=self.output_size[0], height=self.output_size[1], maskthreshold=self.maskthreshold)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        if self.np_array:
            return image
        else:
            image = Image.fromarray(image)
            return image
    
    def call2(self, image, thresh=50):
        image = np.asarray(image)
        (image, _, _) = crop_and_resize(
            image, width=self.output_size[0], height=self.output_size[1], maskthreshold=thresh)
        if self.np_array:
            return image
        else:
            image = Image.fromarray(image)
            return image

    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class AlbumentationsWrapper(object):
    """Wrapper for the albumentations transforms so that it can be used with 
    pytorch transformations.

    Args:
        albs (ablumentations Compose class): a set of albumentations transforms
        image (boolean): should it return a PIL image or stick to numpy array
    """

    def __init__(self, albs, image=False):
        self.albs = albs

    def __call__(self, image):
        if image:
            sample = {'image': np.asarray(image), 'label':0}
        else:
            sample = {'image':image, 'label':0}
        image = self.albs(**sample)
        image = image["image"]
        image = Image.fromarray(image)

        return image

