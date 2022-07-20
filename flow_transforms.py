from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''



class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array2= np.float32(array) 
        array2 = np.transpose(array2, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array2)
        # put it from HWC to CHW format
        return tensor.float()
class FlowToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array2= np.float32(array)   
        #array2 = np.transpose(array2, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array2)
        # put it from HWC to CHW format
        return tensor.float()
