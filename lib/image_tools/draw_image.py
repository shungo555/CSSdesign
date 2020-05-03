import numpy as np
import math
from PIL import Image
import argparse
import sys
import matplotlib.pyplot as plt


sys.path.append('../')
sys.path.append('../../')
from lib.image_tools.image_process import gamma_correction

def imwrite(data, title):
    """image write

	Parameters
	----------
	data : numpy.array (px, py, 3)
		image data array [0, 1]
	title : string
		image title
	"""
    data = np.clip(data, 0, 1)
    img = np.uint8(data * 255)
    Image.fromarray(img).save(title)  


def imwrite255(data, title):
    """image write

    Parameters
    ----------
    data : numpy.array (px, py, 3)
        image data array [0, 255]
    title : string
        image title
    """
    data = np.clip(data, 0, 255)
    img = np.uint8(data)
    Image.fromarray(img).save(title)  


def imwrite_gray(data, title):
    """image write

    Parameters
    ----------
    data : numpy.array (px, py, 1)
        image data array [0, 1]
    title : string
        image title
    """
    data = np.clip(data, 0, 1)
    img = np.uint8(data * 255)
    gray_img = np.concatenate([img,img,img], axis=2)
    Image.fromarray(gray_img).save(title)


def imwrite_gray255(data, title):
    """image write

    Parameters
    ----------
    data : numpy.array (px, py, 1)
        image data array [0, 255]
    title : string
        image title
    """
    data = np.clip(data, 0, 255)
    img = np.uint8(data)
    gray_img = np.concatenate([img,img,img], axis=2)
    Image.fromarray(gray_img).save(title)


def image_hist(data, title):
    fig = plt.figure()
    labels = ['red', 'green', 'blue']
    plt.hist([np.reshape(data[:,:,0],data.shape[0]*data.shape[1]), np.reshape(data[:,:,1],data.shape[0]*data.shape[1]), np.reshape(data[:,:,2],data.shape[0]*data.shape[1])] ,label=labels, bins=50, color=["red","green","blue"], range=[0,1], density=True)
    plt.savefig(title)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input', dest = 'input', default = 'results/', help = 'make input directory')
    parser.add_argument('--output', dest = 'output', default = 'results/', help = 'make output directory')
    parser.add_argument('--title', dest = 'title', default = 'rgb', help = 'image name')
    parser.add_argument('--gamma', dest='gamma', action='store_true', help='gamma correction')
    args = parser.parse_args()
    data = np.load(args.input)
    gamma_data = gamma_correction(data)
    for i in range(data.shape[0]):
        imwrite(data[i,:,:,:], args.output + args.title + str(i) + '.png')
        imwrite(gamma_data[i,:,:,:], args.output + 'gamma_' + args.title + str(i) + '.png')
        image_hist(np.reshape(data[i,:,:,:],(data.shape[1],data.shape[2],data.shape[3])), args.output + args.title + '_hist' + str(i) + '.png')