import numpy as np
import math
from keras import backend as K
K.set_image_data_format('channels_last')
import argparse
import csv


def np_image_cpsnr(y_true, y_pred):
    """calculate cPSNR
    
    Parameters
    ----------
    y_true : numpy.array(px, py, 3)
        true image
    y_pred : numpy.array(px, py, 3)
        predict image
    
    Returns
    -------
    numpy.array(1)
        cPSNR value
    """
    cpsnr = -10 * np.log10(np.mean(np.square(y_pred - y_true), axis=(0, 1, 2)))
    return cpsnr


def np_image_rmse(y_true, y_pred):
    """calculate RMSE
    
    Parameters
    ----------
    y_true : numpy.array(px, py, 3)
        true image
    y_pred : numpy.array(px, py, 3)
        predict image
    
    Returns
    -------
    numpy.array(1)
        RMSE value
    """
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=(0, 1, 2))) * 255
    

def np_cpsnr(y_true, y_pred):
    """calculate cPSNR
    
    Parameters
    ----------
    y_true : numpy.array(batch, px, py, 3)
        true data
    y_pred : numpy.array(batch, px, py, 3)
        predict data
    
    Returns
    -------
    numpy.array(batch size)
        cPSNR value
    """
    cpsnr = -10 * np.log10(np.mean(np.square(y_pred - y_true), axis=(1, 2, 3)))
    return cpsnr


def np_psnr(y_true, y_pred):
    """calculate PSNR

    Parameters
    ----------
    y_true : numpy.array(batch, px, py, 3)
        true data
    y_pred : numpy.array(batch, px, py, 3)
        predict data

    Returns
    -------
    numpy.array(batch size, 3)
        PSNR value
    """
    psnr = -10 * np.log10(np.mean(np.square(y_pred - y_true), axis=(1, 2)))
    return psnr


def np_rmse255(y_true, y_pred):
    """rmse (8bit)
    
    Parameters
    ----------
    y_true : numpy.array(batch, px, py, 3)
        true data
    y_pred : numpy.array(batch, px, py, 3)
        predict data
    
    Returns
    -------
    numpy.array(batch size)
        RMSE value (8bit)
    """
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=(1,2,3))) * 255


def mean_cpsnr(y_true, y_pred):
    """mean cPSNR
    
    Parameters
    ----------
    y_true : array (keras)
        true data
    y_pred : array (keras)
        predict data
    
    Returns
    -------
    array(1) (keras)
        cPSNR value (8bit)
    """
    cpsnr = -10 * K.log( K.mean(K.square(y_pred - y_true), axis=(1,2,3)) ) /math.log(10.0)
    return K.mean( cpsnr )


def mean_rmse255(y_true, y_pred):
    """mean rmse 255 (keras)
    
    Parameters
    ----------
    y_true : array (keras)
        true data
    y_pred : array (keras)
        predict data
    
    Returns
    -------
    array(1) (keras)
        RMSE value (8bit)
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true))) * 255


def main(args):
    data1 = np.load(args.input1)
    data2 = np.load(args.input2)
    YBorder = args.border
    if not YBorder==0:
        val_cpsnr = np_cpsnr(data1[:,YBorder:-YBorder,YBorder:-YBorder,:], data2)
        val_rmse255 = np_rmse255(data1[:,YBorder:-YBorder,YBorder:-YBorder,:], data2)
        val_psnr = np_psnr(data1[:,YBorder:-YBorder,YBorder:-YBorder,:], data2)
    else:
        val_cpsnr = np_cpsnr(data1, data2)
        val_rmse255 = np_rmse255(data1, data2)
        val_psnr = np_psnr(data1, data2)
    print(val_cpsnr)
    # print(val_rmse255)
    # print(val_psnr)    

    # output
    fcsv = open(args.output, 'w')
    writer = csv.writer(fcsv, lineterminator='\n')
    writer.writerow(['cpsnr', 'rmse255', 'psnrR', 'psnrG', 'psnrB'])
    fcsv.flush()
    for i in range(data1.shape[0]):
        writer.writerow([val_cpsnr[i], val_rmse255[i], val_psnr[i, 0], val_psnr[i, 1], val_psnr[i, 2]])
    fcsv.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input1', dest = 'input1', default = 'file1', help = 'make input1 file name')
    parser.add_argument('--input2', dest = 'input2', default = 'file2', help = 'make input2 file name')
    parser.add_argument('--output', dest = 'output', default = 'filename.csv', help = 'make output file name')
    parser.add_argument('--border', type=int, default=0, help='border')
    args = parser.parse_args()
    main(args)