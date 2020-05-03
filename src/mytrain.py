#!/usr/bin/env python

# -*- coding: utf-8 -*-

import pickle
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import argparse
from datetime import datetime as dt
import threading
import sys
import os
import math
import numpy as np
import build_my_model
import build_model
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.optimizers import Adam
from mydata import Gen
from keras import backend as K
K.set_image_data_format('channels_last')

# import data
# from mydata import gen1

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.image_tools.cfa import cfa_bayer
from lib.image_tools.evaluation import np_cpsnr, np_rmse255, mean_cpsnr, mean_rmse255
from lib.image_tools.draw_image import imwrite, imwrite_gray
from lib.load_dataset.load_environment_data import get_illuminant
from lib.load_dataset.load_camera_data import get_camera_sensitivity
from lib.train_tools.plot_tool import plot_all_log, plot_sensitivity

# def output_config(test_data_num, smoothness, nl_max255, batch_size, patch_size):
#     strings = [
#         '###config###',
#         'test data number: ' + str(test_data_num),
#         'smoothness: ' + str(smoothness),
#         'max noise level (255): ' + str(nl_max255),
#         'training batch size: ' + str(batch_size),
#         'training patch size: ' + str(patch_size),
#     ]
#     with open(TRAIN_RESULT_PATH + 'config.txt', mode='w', encoding="utf-8") as f:
#         f.write("\n".join(strings))
#         f.write("\n")
#     f.close


def train(model, gen_class, val_gen_class, epochs, steps_per_epoch = 100):
    # Initialize parameters
    max_val_mean_cpsnr = 0
    weight_array = np.zeros((int(epochs), model.get_weights()[1][0][0].shape[0],model.get_weights()[1][0][0].shape[1]))
    # Generator
    gen = gen_class.generator()

    val_gen = val_gen_class.generator()
    val_data = val_gen.__next__()

    # Read test data
    Xtest = val_data[0]
    Ytest = val_data[1]
    test_noise = val_data[2]
    gain_val = val_data[3]

    # CSV setting
    line = 'timestamp, epoch'
    for metric in model.metrics_names:
        line += ', ' + metric
    line += ', val_rmse255, val_mean_cpsnr'
    print(line)
    cpsnr_log = []
    fcsv = open('data.csv', 'w')
    fcsv.write(line +'\n')
    fcsv.flush()

    # Training
    for epoch in range(epochs):
        SH = [0] * len(model.metrics_names)
        for step in range(steps_per_epoch):
            X, Y, noise, gains = gen.__next__()
            h = model.train_on_batch([X, noise, gains, 1 / gains], Y)
            SH = [_SH + _h for _SH, _h in zip(SH, h)]
            new_sens = model.get_weights()[1][0][0]
            gen_class.update_sens(new_sens)

        # predict and evaluate
        val_gen_class.update_sens(new_sens)
        gain_val = val_gen.__next__()[3]
        Ypred = model.predict([Xtest, test_noise, gain_val, 1 / gain_val])
        val_rmse255 = np.mean(np_rmse255(Ypred, Ytest))
        val_mean_cpsnr = np.mean(np_cpsnr(Ypred, Ytest))

        # print log
        SH = [_SH / steps_per_epoch for _SH in SH]
        timestamp = dt.now().strftime('%Y/%m/%d %H:%M:%S')
        line = f'{timestamp}, {epoch}'
        for _SH in SH:
            line += ', {}'.format(_SH)
        line += f', {val_rmse255}, {val_mean_cpsnr}'

        print(line)

        # save log
        fcsv.write(line+'\n')
        fcsv.flush()
        cpsnr_log.append(val_mean_cpsnr)
        weight_array[epoch,:,:] = model.get_weights()[1][0][0]

        # save weight
        if(max_val_mean_cpsnr < val_mean_cpsnr):
            max_val_mean_cpsnr = val_mean_cpsnr
            np.save('Ypred.npy', Ypred)
            model.save_weights('img_best.hdf5')

    # Save final weight
    model.save_weights('model_weight.hdf5')
    fcsv.close()
    opt_sens = model.get_weights()[1][0][0]
    plot_sensitivity(opt_sens, [400, 700], 'optimal_sensitivity.png')
    np.save('sens_log.npy', weight_array)
    print('Finish')


def main(args):
    # GPU setting
    if args.gpu:
        # For fujin server
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Using gpu')

    # initial camera sensitivity
    camera_name = args.camera
    # camera_name = "Canon20D"
    # camera_name = "Canon300D"
    # camera_name = "NikonD40"
    # camera_name = "OlympusE-PL2"


    print('Load dataset : ' + args.dataset)
    if args.dataset == 'cave':
        from lib.load_dataset.load_cave_data import get_hsi, get_srgb, get_srgb_gain, get_val_srgb
        # wavelengt range
        wavelength_range = [400, 700]
    elif args.dataset == 'tokyotech':
        from lib.load_dataset.load_tokyotech_data import get_hsi, get_srgb, get_srgb_gain, get_val_srgb
        # wavelengt range
        wavelength_range = [420, 720]
    else:
        print('dataset error')
        sys.exit(1)

    # Load hyperspectral images
    hsi = get_hsi()
    data_num = hsi.shape[0]
    hsi_bandwidth = hsi.shape[3]

    # Load sRGB and cRGB images
    srgb = get_srgb()

    # Load initial camera sensitivity
    sens = get_camera_sensitivity(camera_name=camera_name, wavelength_range=wavelength_range)
    rgb_bandwidth = sens.shape[1]
    sensitivity = np.zeros((1, 1, hsi_bandwidth, rgb_bandwidth))
    sensitivity[0][0] = sens

    # plot sensitivity
    plot_sensitivity(sens, wavelength_range, args.output + 'sensitivity.png')

    # Load illuminants
    Ls = np.zeros((1, 1, hsi_bandwidth, hsi_bandwidth))
    Ls[0][0] = get_illuminant(wavelength_range)

    # Normalize each image
    g_ts = get_srgb_gain()
    for i in range(data_num):
        hsi[i, :, :, :] *= g_ts[i]

    # set ground truth
    print('Ground Truth is ' + args.gt)
    if (args.gt == 'srgb'):
        ground_truth = srgb
        with open('D:/workspace/大学/奥富田中研究室/program/dataset/cave_hsi/val_data/srgb/CSS_change/val_data_noise0.pickle', mode='rb') as fi:
            val_data = pickle.load(fi)


    # Data details
    YBorder = 1
    nl_max255 = args.noise
    nl_max = args.noise / 255.0
    batch_size = 32
    ps = 128
    patch_size = [ps + 2 * YBorder, ps + 2 * YBorder]
    nl_range = [0, nl_max]
    
    # Train parameters
    test_data_num = 8
    traindata_num = data_num - test_data_num
    # smoothness = 1E-6
    smoothness = 0
    skip_mixed = True
    Mcc = None
    epochs = args.epochs

    # # Save config
    # output_config(test_data_num, smoothness, nl_max255, batch_size, ps)

    if args.validation:
        # Make validation data
        print('make validation data')
        gen_class = Gen(hsi[-test_data_num:, :, :, :], ground_truth[-test_data_num:, :, :, :], batch_size, patch_size, Ls[0][0], sens, nl_range = nl_range, YBorder = YBorder)   
        gen_class.seeds(0, 1)
        gen = gen_class.generator()
        val_data = gen.__next__()

        with open(args.output +  '/val_data_noise' + str(args.noise) + '.pickle', mode='wb') as fo:
            pickle.dump(val_data, fo) 
    else:
        # Train data
        gen_class = Gen(hsi[:-test_data_num, :, :, :], ground_truth[:-test_data_num, :, :, :],
                            batch_size, patch_size, Ls[0][0], sens, nl_range=nl_range, YBorder=YBorder)
        gen_class.seeds(0, 1)

        # Define model
        encoder= build_my_model.MyEncoder(initial_sensitivity=sensitivity, Ls=Ls, smoothness=smoothness, input_shape=(
            patch_size[0], patch_size[1], hsi_bandwidth), trainable=True).my_encoder()
        decoder = build_model.WiG_sub(input_shape=(
            patch_size[0], patch_size[1], 1), nb_features=128, Mcc=Mcc, skip_mixed=skip_mixed)

        gain_class = build_my_model.GainLayer(input_shape=(patch_size[0]-2*YBorder, patch_size[1]-2*YBorder , 3))
        gain_multi_model = gain_class.gain_multi_layer()
        gain_model = gain_class.gain_define_layer()

        model_inputs = [encoder.inputs[0], encoder.inputs[1], encoder.inputs[2], gain_model.inputs[0]]
        model_outputs = gain_multi_model([decoder(encoder.outputs[0]), gain_model.outputs[0]])
        model = Model(inputs = model_inputs, outputs = model_outputs)

        opt = Adam(clipvalue=0.5)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=[mean_rmse255, mean_cpsnr])

        # Output model structure
        model.summary()
        # plot_model(model, to_file = args.output + 'model.png')

        # Load model weight (if exists)
        if (args.weight != None):
            print('Load' + args.weight)
            model.load_weights(args.weight)

        # Define output folder
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        os.chdir(args.output)


        val_gen_class = Gen(hsi[-test_data_num:, :, :, :], ground_truth[-test_data_num:, :, :, :], batch_size, patch_size, Ls[0][0], sens, nl_range = nl_range, YBorder = YBorder)   
        val_gen_class.seeds(0, 1)

        # Train
        train(model, gen_class, val_gen_class, epochs)

        plot_all_log(args.output + 'data.csv', args.output)




if __name__ == "__main__":
    # arguments configuration
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='use the GPU for processing. (default: True)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs. (default: 100)')
    parser.add_argument('--noise', type=int, default=0,
                        help='define test noise level (8bit). (default: 0)')
    parser.add_argument('--output', dest='output',
                        default='results/', help='output directory (necessary)')
    parser.add_argument('--weight', dest='weight',
                        help='load existing weight file (default: None)')
    parser.add_argument('--gt', dest='gt', default='srgb',
                        help='set ground truth as srgb or crgb (default: srgb)')
    parser.add_argument('--dataset', dest='dataset', default='cave',
                        help='set dataset cave or tokyotech (default: cave)')
    parser.add_argument('--camera', dest = 'camera',  default='Canon20D',
                        help = 'if you use other cameras, set camera name (default: Canon20D)')
    parser.add_argument('--validation', dest = 'validation', action = 'store_true', 
                        help = 'if you make validation data, set True (default: False)')
    parser.set_defaults(gpu = True)
    parser.set_defaults(weight = None)
    parser.set_defaults(validation = False)
    args = parser.parse_args()
    main(args)
