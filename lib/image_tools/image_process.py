import numpy as np
import math
from PIL import Image
import argparse


def gamma_correction(data):
    """gamma correction Y=X^(1/2.2)

    Parameters
    ----------
    data : numpy array (px, py, 3)
        image data array [0, 1]
    """
    gamma = 2.2
    data = data.clip(min=0)
    gamma_data = np.power(data, 1 / gamma)
    return gamma_data


def create_crgb(hsi, sens, Ls):
    """create crgb data

    Parameters
    ----------
    hsi : numpy.array(px, py, hsi_bands)
        hsi data
    sens : numpy.array(hsi_bands, rgb_bands)
        sensitivity
    Ls : numpy.array(hsi_bands, hsi_bands)
        illuminants data

    Returns
    -------
    numpy.array(px, py, 3)
        crgb data
    """
    data = np.reshape(hsi[:, :, :], (hsi.shape[0]
                                     * hsi.shape[1], hsi.shape[2]))
    crgb = sens.T @ Ls @ data.T
    crgb = np.transpose(np.reshape(
        crgb, (3, hsi.shape[0], hsi.shape[1])), (1, 2, 0))
    return crgb


def create_3bands_raw(raw, cfa):
    """create 3-bands raw data from 1-band raw data
    
    Parameters
    ----------
    raw : numpy.array(batch_size, px, py, 1)
        1-band raw data
    cfa : numpy.array(1, px, py, rgb_bands)
        cfa array

    Returns
    -------
    numpy.array(batch_size, px, py, 3)
        raw data
    """
    raw3 = np.zeros((raw.shape[0], raw.shape[1], raw.shape[2], 3))
    for i in range(raw.shape[0]):
        for band in range(3):
            raw3[i, :, :, band] = raw[i, :, :, 0] * cfa[0, :, :, band]
    return raw3


def create_raw(hsi, sens, Ls, cfa):
    """create raw data

    Parameters
    ----------
    hsi : numpy.array(px, py, hsi_bands)
        hsi data
    sens : numpy.array(hsi_bands, rgb_bands)
        sensitivity
    Ls : numpy.array(hsi_bands, hsi_bands)
        illuminants data
    cfa : numpy.array(1, px, py, rgb_bands)
        cfa array

    Returns
    -------
    numpy.array(px, py, 1)
        raw data
    """
    crgb = create_crgb(hsi, sens, Ls)
    raw = crgb[:, :, 0] * cfa[0, :, :, 0] + crgb[:, :, 1] * \
        cfa[0, :, :, 1] + crgb[:, :, 2] * cfa[0, :, :, 2]
    raw = np.reshape(raw, (raw.shape[0], raw.shape[1], 1))
    return raw


def calculate_gain(hsi, sens, Ls, gain_max=0.95):
    """calculate gain s.t. max(raw)=gain_max (defalut, 0.95)

    Parameters
    ----------
    hsi : numpy.array(px, py, hsi_bands)
        hsi data
    sens : numpy.array(hsi_bands, rgb_bands)
        sensitivity
    Ls : numpy.array(hsi_bands, hsi_bands)
        illuminants data
    cfa : numpy.array(1, px, py, rgb_bands)
        cfa array
    gain_max : float, optional
        max(raw)=gain_max, by default 0.95

    Returns
    -------
    numpy.array(1)
        gain
    """
    data = np.reshape(hsi[:, :, :], (hsi.shape[0]
                                     * hsi.shape[1], hsi.shape[2]))
    crgb = sens.T @ Ls @ data.T
    crgb = np.transpose(np.reshape(
        crgb, (3, hsi.shape[0], hsi.shape[1])), (1, 2, 0))
    gain = gain_max / np.max(crgb)
    return gain


def calculate_raw_gain(hsi, sens, Ls, cfa, gain_max=0.95):
    """calculate gain s.t. max(raw)=gain_max (defalut, 0.95)

    Parameters
    ----------
    hsi : numpy.array(px, py, hsi_bands)
        hsi data
    sens : numpy.array(hsi_bands, rgb_bands)
        sensitivity
    Ls : numpy.array(hsi_bands, hsi_bands)
        illuminants data
    cfa : numpy.array(1, px, py, rgb_bands)
        cfa array
    gain_max : float, optional
        max(raw)=gain_max, by default 0.95

    Returns
    -------
    numpy.array(1)
        gain
    """
    data = np.reshape(hsi[:, :, :], (hsi.shape[0]
                                     * hsi.shape[1], hsi.shape[2]))
    crgb = sens.T @ Ls @ data.T
    crgb = np.transpose(np.reshape(
        crgb, (3, hsi.shape[0], hsi.shape[1])), (1, 2, 0))
    raw = crgb[:, :, 0] * cfa[0, :, :, 0] + crgb[:, :, 1] * cfa[0, :, :, 1] + crgb[:, :, 2] * cfa[0, :, :, 2]
    gain = gain_max / np.max(raw)
    return gain