# -*- coding: utf-8 -*-
"""
@author: Masaki
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
import sys
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')

sys.path.append('../')
sys.path.append('../../')
from lib.load_dataset.load_illuminant_data import get_illuminant
from lib.load_dataset.load_camera_data import get_camera_sensitivity
from lib.load_dataset.load_environment_data import get_H
from lib.image_tools.draw_image import imwrite
from lib.image_tools.image_process import gamma_correction


def main(args):
    print('Load dataset : ' + args.dataset)
    if args.dataset == 'cave':
        from lib.load_dataset.load_cave_data import get_hsi
        # wavelengt range
        wavelength_range = [400, 700]
    elif args.dataset == 'tokyotech':
        from lib.load_dataset.load_tokyotech_data import get_hsi
        # wavelengt range
        wavelength_range = [420, 720]
    else:
        print('dataset error')
        sys.exit(1)

    Ls = get_illuminant(illuminant_name="D65", wavelength_range=wavelength_range)
    print(Ls)

    # load HSI data
    hsi = get_hsi()
    data_num = hsi.shape[0]
    data = np.reshape(
        hsi, (hsi.shape[0], hsi.shape[1] * hsi.shape[2], hsi.shape[3]))

    print('Ground Truth : ' + args.gt)
    # load sensitivity
    if (args.gt == 'srgb'):
        sens = get_H(wavelength_range=wavelength_range)
    elif (args.gt == 'crgb'):
        sens = get_camera_sensitivity(camera_name="Canon20D", wavelength_range=wavelength_range)

    # get RGB gain(for normalize)
    alpha = 0.95
    g_t = np.ones((hsi.shape[0], 1))

    # make srgb data
    rgb_data = np.zeros((hsi.shape[0], hsi.shape[1] * hsi.shape[2], 3))
    gamma_rgb_data = np.zeros((hsi.shape[0], hsi.shape[1] * hsi.shape[2], 3))

    for i in range(data_num):
        rgb_data[i, :, :] = np.dot(sens, np.dot(Ls, data[i, :, :].T)).T
        g_t[i, 0] = alpha / np.amax(rgb_data[i, :, :])
        rgb_data[i, :, :] *= g_t[i, 0]
        gamma_rgb_data[i, :, :] = gamma_correction(rgb_data[i, :, :])

    rgb = np.reshape(rgb_data, (hsi.shape[0], hsi.shape[1], hsi.shape[2], 3))
    gamma_rgb = np.reshape(
        gamma_rgb_data, (hsi.shape[0], hsi.shape[1], hsi.shape[2], 3))

    np.save(args.output + 'gt_' + args.gt + '.npy', g_t)
    np.save(args.output + args.dataset + '_' + args.gt + '.npy', rgb)

    if (args.gt == 'srgb'):
        np.savez(args.output + args.dataset + '_' + args.gt + '.npz', srgb=rgb)
        for i in range(data_num):
            imwrite(rgb[i, :, :, :], args.output +
                    'img/sRGB_' + str(i) + '.png')
            imwrite(gamma_rgb[i, :, :, :], args.output +
                    'img/gamma_sRGB_' + str(i) + '.png')

    elif(args.gt == 'crgb'):
        np.savez(args.output + args.dataset + '_' + args.gt + '.npz', crgb=rgb)
        for i in range(data_num):
            imwrite(rgb[i, :, :, :], args.output +
                    'img/cRGB_' + str(i) + '.png')

    print('Finish')
    
if __name__ == "__main__":
    # arguments configuration
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--gt', dest='gt', default='srgb',
                        help='set ground truth srgb or crgb (default: srgb)')
    parser.add_argument('--output', dest='output',
                        default='output', help='output dir')
    parser.add_argument('--dataset', dest='dataset', default='cave',
                        help='set dataset cave or tokyotech (default: cave)')
    args = parser.parse_args()
    main(args)
