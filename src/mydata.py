#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.utils import Sequence

import shutil
import requests
import zipfile
import os
import sys
import glob
import random
from PIL import Image
import numpy as np
import scipy
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.image_tools.cfa import cfa_bayer
from lib.image_tools.image_process import create_raw, calculate_gain

class Gen():
    """generator class
    """

    def __init__(self, X, Y, batch_size, patch_size, Ls, sens, nl_range=[0, 30], YBorder=1, data_augmentation=True):
        """initialize generator

        Parameters
        ----------
        X : numpy.array(batch_size, px, py, hsi_bands)
            HSI data
        Y : numpy.array(batch_size, px, py, rgb_bands)
            True data
        batch_size : int
            batch size
        patch_size : list[px, py]
            patch size
        Ls : numpy.array(hsi_bands, hsi_bands)
            illuminant data
        sens : numpy.array(hsi_bands, rgb_bands)
            sensitivity data
        nl_range : list, optional
            noise range, by default [0, 30]
        YBorder : int, optional
            Yborder, by default 1
        data_augmentation : bool, optional
            data augmentation flag, by default True
        """
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.Ls = Ls
        self.sens = sens
        self.nl_range = nl_range
        self.YBorder = YBorder
        self.data_augmentation = data_augmentation
        self.gains = np.zeros([self.X.shape[0], 1])
        for i in range(self.X.shape[0]):
            self.gains[i, 0] = calculate_gain(X[i, :, :, :], self.sens, self.Ls)


    def seeds(self, random_seed=0, np_random_seed=0):
        """define random seed value

        Parameters
        ----------
        random_seed : int, optional
            random seed, by default 0
        np_random_seed : int, optional
            numpy seed, by default 0
        """
        random.seed(random_seed)
        np.random.seed(np_random_seed)


    def generator(self):
        """generator function

        Yields
        -------
        mosaic1, yy, noise
        """
        x = np.zeros(
            (self.batch_size, self.patch_size[0], self.patch_size[1], self.X.shape[3]))
        y = np.zeros(
            (self.batch_size, self.patch_size[0], self.patch_size[1], self.Y.shape[3]))
        noise = np.zeros(
            (self.batch_size, self.patch_size[0], self.patch_size[1], 1))
        posrange = [self.X.shape[1] - self.patch_size[0],
                    self.X.shape[2] - self.patch_size[1]]
        gains = np.zeros((self.batch_size, 1))

        while(True):
            for i in range(self.batch_size):
                k = random.randrange(0, self.X.shape[0])
                p = [random.randrange(0, posrange[0]),
                     random.randrange(0, posrange[1])]

                tx = self.X[k, p[0]: p[0] + self.patch_size[0],
                            p[1]: p[1] + self.patch_size[1], :]
                ty = self.Y[k, p[0]: p[0] + self.patch_size[0],
                            p[1]: p[1] + self.patch_size[1], :]

                if(self.data_augmentation):
                    b = random.randrange(0, 8)

                    if((b & 0b100) > 0):  # transpose
                        tx = np.swapaxes(tx, 0, 1)
                        ty = np.swapaxes(ty, 0, 1)

                    if((b & 0b010) > 0):  # mirror
                        tx = np.flip(tx, 1)
                        ty = np.flip(ty, 1)

                    if((b & 0b001) > 0):  # flip
                        tx = np.flip(tx, 0)
                        ty = np.flip(ty, 0)

                x[i, :, :, :] = tx
                y[i, :, :, :] = ty

                if(self.YBorder > 0):
                    yy = y[:, self.YBorder:-
                           (self.YBorder), self.YBorder:-(self.YBorder), :]
                nl = np.random.rand() * \
                    (self.nl_range[1] - self.nl_range[0]) + self.nl_range[0]
                noise[i, :, :, :] = np.random.normal(
                    scale=nl, size=(self.patch_size[0], self.patch_size[1], 1))
                gains[i, 0] = self.gains[k, 0]
            yield x, yy, noise, gains


    def update_sens(self, new_sens):
        """update sensitivity

        Parameters
        ----------
        new_sens : numpy.array(hsi_bands, rgb_bands)
            update sensitivity
        """
        self.sens = new_sens
        for i in range(self.X.shape[0]):
            self.gains[i, 0] = calculate_gain(self.X[i, :, :, :], self.sens, self.Ls)