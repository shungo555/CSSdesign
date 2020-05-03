from keras import backend as K
K.set_image_data_format('channels_last')

import build_model, build_my_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import Model, Sequential

import numpy as np
import math
import os
import sys
import threading
from datetime import datetime as dt
import argparse
import scipy.io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import csv
import random
import pickle
from mydata_numpy import gen_numpy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.image_tools.cfa import cfa_bayer
from lib.image_tools.evaluation import np_cpsnr, np_rmse255, mean_cpsnr, mean_rmse255
from lib.image_tools.draw_image import imwrite, imwrite_gray
from lib.image_tools.image_process import create_raw, calculate_gain, gamma_correction, create_3bands_raw
from lib.load_dataset.load_environment_data import get_illuminant
from lib.load_dataset.load_camera_data import get_camera_sensitivity
from lib.train_tools.plot_tool import plot_sensitivity

def test(model, hsi, ground_truth, noise, gains, output_dir):
    """test model
    
    Parameters
    ----------
    model : keras model
        trained model
    raw : numpy.array(batch_size, px, py, 1)
        raw data
    ground_truth : numpy.array(batch_size, px, py, 3)
        ground truth data
    gains : numpy.array(batch_size)
        gain data
    """
    # initialize
    Xtest = hsi
    Ytrue = ground_truth
    Ypred = np.zeros_like(Ytrue)
    data_num = Xtest.shape[0]

    # test images
    for i in range(hsi.shape[0]):
        Xtest_i = np.reshape(Xtest[i, :, :, :], (1, Xtest.shape[1], Xtest.shape[2], Xtest.shape[3]))
        noise_i = np.reshape(noise[i, :, :, :], (1, noise.shape[1], noise.shape[2], noise.shape[3]))
        gains_i = np.reshape(gains[i, 0], (1, 1))
        Ypred[i, :, :, :] = model.predict([Xtest_i, noise_i, gains_i, 1 / gains_i])

    # evaluation
    val_cpsnr = np_cpsnr(Ytrue, Ypred)
    val_rmse255 = np_rmse255(Ytrue, Ypred)
    
    # print(error_rate(Ytrue, Ypred))
    print('average:' + str(np.mean(val_rmse255)))
    print('average:' + str(np.mean(val_cpsnr)))

    # output
    fcsv = open(output_dir + '/data.csv', 'w')
    writer = csv.writer(fcsv, lineterminator='\n')
    writer.writerow(['cpsnr', 'rmse255'])
    fcsv.flush()
    for i in range(data_num):
        writer.writerow([val_cpsnr[i], val_rmse255[i]])
    fcsv.close()

    gamma_Ypred = gamma_correction(Ypred)

    # save
    np.save(output_dir + '/Ypred.npy', Ypred)

    if args.gt == 'srgb':
        for i in range(data_num):
            imwrite(Ypred[i, :, :, :], output_dir + 'img/Ypred' + str(i) + '.png')
            imwrite(Ytrue[i, :, :, :], output_dir + 'img/sRGB' + str(i) + '.png')
            imwrite(gamma_Ypred[i, :, :, :], output_dir + 'img/gamma_Ypred' + str(i) + '.png')
    elif args.gt == 'crgb':
        for i in range(data_num):
            imwrite(Ypred[i, :, :, :], output_dir + 'img/Ypred' + str(i) + '.png')
            imwrite(Ytrue[i, :, :, :], output_dir + 'img/cRGB' + str(i) + '.png')


def main(args):
    # GPU setting
    if args.gpu:
        # For fujin server
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Using gpu')

    camera_name = args.camera
    # camera_name = "NikonD40"

    # Define dataset
    print('Load dataset : ' + args.dataset)
    if args.dataset == 'cave':
        from lib.load_dataset.load_cave_data import get_hsi, get_crgb, get_srgb, get_crgb_gain, get_srgb_gain
        # wavelengt range
        wavelength_range = [400, 700]
    elif args.dataset == 'tokyotech':
        from lib.load_dataset.load_tokyotech_data import get_hsi, get_crgb, get_srgb, get_crgb_gain, get_srgb_gain
        # wavelengt range
        wavelength_range = [420, 720]
    else:
        print('dataset error')
        sys.exit(1)

    # Load hyperspectral images
    hsi = get_hsi()
    data_num = hsi.shape[0]
    hsi_bandwidth = hsi.shape[3]

    # Load sRGB images
    srgb = get_srgb()

    # Load cRGB images
    crgb = get_crgb()

    # Load initial camera sensitivity
    sens = get_camera_sensitivity(camera_name, wavelength_range)
    rgb_bandwidth = sens.shape[1]
    sensitivity = np.zeros((1, 1, hsi_bandwidth, rgb_bandwidth))
    sensitivity[0][0] = sens

    # Load illuminants
    L = get_illuminant(wavelength_range)
    Ls = np.zeros((1, 1, hsi_bandwidth, hsi_bandwidth))
    Ls[0][0] = L

    # Load gains
    g_ts = get_srgb_gain()

    YBorder = 1

    # set ground truth
    print('Ground Truth is ' + args.gt)
    if (args.gt == 'srgb'):
        ground_truth = srgb[:, YBorder:-YBorder, YBorder:-YBorder, :]
    elif (args.gt == 'crgb'):
        ground_truth = crgb[:, YBorder:-YBorder, YBorder:-YBorder, :]

    # Normalize each image
    for i in range(data_num):
        hsi[i, :, :, :] *= g_ts[i]
    
    # Define cfa
    cfa = cfa_bayer([hsi.shape[1], hsi.shape[2]])

    # Test parameters
    noise_level = args.noise / 255.0
    smoothness = 1E-8
    Mcc = None
    skip_mixed = True

    # Define model
    encoder= build_my_model.MyEncoder(initial_sensitivity=sensitivity, Ls=Ls, smoothness=smoothness, input_shape=(
        hsi.shape[1], hsi.shape[2], hsi_bandwidth), trainable=False).my_encoder()
    decoder = build_model.WiG_sub(input_shape=(
        hsi.shape[1], hsi.shape[2], 1), nb_features=128, Mcc=Mcc, skip_mixed=skip_mixed)

    gain_class = build_my_model.GainLayer(input_shape=(hsi.shape[1] - 2*YBorder, hsi.shape[2] - 2*YBorder , 3))
    gain_multi_model = gain_class.gain_multi_layer()
    gain_model = gain_class.gain_define_layer()

    model_inputs = [encoder.inputs[0],encoder.inputs[1],encoder.inputs[2], gain_model.inputs[0]]
    model_outputs = gain_multi_model([decoder(encoder.outputs[0]), gain_model.outputs[0]])
    model = Model(inputs = model_inputs, outputs = model_outputs)

    # Load weight
    if (args.weight != None):
        print('Load:' + args.weight)
        model.load_weights(args.weight)

    sens = model.get_weights()[1][0][0]
    plot_sensitivity(sens, wavelength_range, args.output + 'optimal_sensitivity.png')

    gains = np.zeros((hsi.shape[0], 1))
    noise = np.zeros((hsi.shape[0], hsi.shape[1], hsi.shape[1], 1))
    for i in range(hsi.shape[0]):
        gains[i, 0] = calculate_gain(hsi[i, :, :, :], sens, Ls[0][0])
        noise[i, :, :, :] = np.random.normal(scale=noise_level, size=(hsi.shape[1], hsi.shape[1], 1))
        
    test(model, hsi, ground_truth, noise, gains, args.output)

    # # output model structure
    model.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='use the GPU for processing. (default: True)')
    parser.add_argument('--weight', dest='weight',
                        help='load weight file (default: None)')
    parser.add_argument('--noise', type=int, default=0,
                        help='define test noise level (8bit). (default: 0)')
    parser.add_argument('--output', dest='output',
                        default='results/', help='make output directory')
    parser.add_argument('--gt', dest='gt', default='srgb', help='srgb or crgb')
    parser.add_argument('--camera', dest = 'camera',  default='Canon20D',
                        help = 'if you use other cameras, set camera name (default: Canon20D)')
    parser.set_defaults(weight=None)
    parser.add_argument('--dataset', dest='dataset', default='cave',
                        help='set dataset cave or tokyotech (default: cave)')
    args = parser.parse_args()
    main(args)
