import numpy as np
import scipy.io
import os
import sys
from PIL import Image
import argparse
import csv
import h5py
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('..\\')
sys.path.append('..\\..\\')
from lib.load_dataset.load_camera_data import get_camera_sensitivity
from lib.load_dataset.load_illuminant_data import get_illuminant
from lib.load_dataset.load_environment_data import get_chart, get_chart_hsi, get_H
from lib.image_tools.image_process import gamma_correction
from lib.image_tools.draw_image import imwrite


def chart_board(C, Ls):    
    """chart_board
    
    Parameters
    ----------
    C : numpy.array(rgb_bands, hsi_bands)
        camera sensitivity data
    Ls : numpy.array(hsi_bands, hsi_bands)
        illuminant data
    
    Returns
    -------
    numpy.array(128, 192, 3)
        chart rgb data
    """
    hsi_R = get_chart_hsi()
    px = hsi_R.shape[0]
    py = hsi_R.shape[1]
    band = hsi_R.shape[2]
    crgb_R0 = C @ Ls @ np.reshape(hsi_R, (px * py, band)).T
    crgb_R = np.reshape(crgb_R0.T, (px, py, 3))
    return crgb_R


def make_M(C, H, Ls):
    """make color correction matrix
    
    Parameters
    ----------
    C : numpy.array(rgb_bands, hsi_bands)
        camera sensitivity data
    Ls : numpy.array(hsi_bands, hsi_bands)
        illuminant data
    Returns
    -------
    numpy.array(rgb_bands, rgb_bands)
        color correction matrix
    """
    # Load data
    R = get_chart()

    # obtain M
    Y_c = C @ Ls @ R
    Y_s = H @ Ls @ R
    invY_c = np.linalg.pinv(Y_c)
    M = np.dot(Y_s, invY_c)

    return M


def color_correct(crgb, C, H, Ls):
    """color correct

    Parameters
    ----------
    crgb : numpy.array(batch_size, px, py, rgb_bands)
        crgb data
    C : numpy.array(rgb_bands, hsi_bands)
        camera sensitivity data
    
    Returns
    -------
    numpy.array(batch_size, px, py, rgb_bands)
        color corrected srgb data
    """
    # Obtain color correction matrix
    M = make_M(C, H, Ls)
    print(M)

    # coorect cRGB to sRGB 
    data_num = crgb.shape[0]
    px = crgb.shape[1]
    py = crgb.shape[2]
    band = crgb.shape[3]    
    correct_data0 = np.dot(M, np.reshape(crgb, (data_num * px * py,band)).T)
    srgb_ref = np.reshape(correct_data0.T, (data_num, px, py, band))
    print(np.mean(srgb_ref))
    return srgb_ref


def main(args):
    # Load dataset
    if args.dataset == 'cave':
        from lib.load_dataset.load_cave_data import get_crgb_gain, get_srgb_gain, get_crgb_gain_s, get_srgb_gain_s, get_chart_crgb_gain, get_chart_srgb_gain
        # wavelengt range
        wavelength_range = [400, 700]
    elif args.dataset == 'tokyotech':
        from lib.load_dataset.load_tokyotech_data import get_crgb_gain, get_srgb_gain, get_crgb_gain_s, get_srgb_gain_s, get_chart_crgb_gain, get_chart_srgb_gain
        # wavelengt range
        wavelength_range = [420, 720]
    else:
        print('dataset error')
        sys.exit(1)

    # Load Camera sensitivity
    camera_name = 'Canon20D'
    C = get_camera_sensitivity(camera_name=camera_name, wavelength_range=wavelength_range).T
    H = get_H(wavelength_range=wavelength_range)
    
    # Load cRGB
    crgb = np.load(args.crgb) * 0.95
    # crgb_gain = get_crgb_gain()    
    # srgb_gain = get_srgb_gain()   
    crgb_gain = np.load(args.gain) 

    illuminant_name = 'D65'
    Ls = get_illuminant(illuminant_name=illuminant_name, wavelength_range=wavelength_range)

    # normalize cRGB
    # for i in range(crgb.shape[0]):
    #     crgb[i,:,:,:] /= crgb_gain[i] 
    #     # crgb[i,:,:,:] *= srgb_gain[i] 

    # color correction
    srgb_ref = color_correct(crgb, C, H, Ls)
    # print(srgb_gain)
    for i in range(srgb_ref.shape[0]):
        print(np.amax(srgb_ref[i,:,:,:]))
        print(crgb_gain[i])
        srgb_ref[i,:,:,:]*=crgb_gain[i]

    np.save(args.output, srgb_ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--crgb', dest = 'crgb', default = 'results/', help = 'make input directory')
    parser.add_argument('--gain', dest = 'gain', default = 'results/', help = 'make input directory')
    parser.add_argument('--output', dest = 'output', default = 'results/', help = 'make output directory')
    parser.add_argument('--dataset', dest='dataset', default='cave', help='set dataset cave or tokyotech (default: cave)')
    args = parser.parse_args()
    main(args)