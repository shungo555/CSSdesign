import bm3d
import numpy as np
import argparse
import os
from joblib import Parallel, delayed
import joblib

def denoise_bm3d(data, noise_level, index):
    print('index : ' + str(index))
    print('noise_level' + str(noise_level))
    d = bm3d.bm3d(data, noise_level)
    return d, index
    

def main(args):
    data = np.load(args.input)
    gain = np.load(args.gain)
    noise_level = args.noise / 255.0
    data_num = data.shape[0]
    processed = joblib.Parallel(n_jobs=-1)([delayed(denoise_bm3d)(data[index,:,:,:], noise_level / gain[index], index) for index in range(data_num)])
    processed.sort(key=lambda x: x[1])
    processed_data = np.array([t[0] for t in processed])
    np.save(args.output, processed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input', dest='input', help='load input file ')
    parser.add_argument('--output', dest = 'output', default = 'results/', help = 'make output directory')
    parser.add_argument('--noise', type = int, default = 0, help = 'define test noise level (8bit). (default: 0)')
    parser.add_argument('--gain', dest='gain', help='load gain file ')
    args = parser.parse_args()
    main(args)