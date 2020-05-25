import numpy as np
import scipy.io
import csv
import pickle


CAVE_DATA_ROOT = 'D:/workspace/大学/奥富田中研究室/program/dataset/cave_hsi/'


def get_hsi():
    """load cave hsi
    
    Returns
    -------
    numpy.array(32, 512, 512, 31)
        hsi data (batch, px, py, bands)
    """    
    print('Load HSI data')
    npz = np.load(CAVE_DATA_ROOT + 'cave_hsi.npz')
    hsi = npz['X']
    return hsi


def get_srgb():
    """load cave sRGB
    
    Returns
    -------
    numpy.array(32, 512, 512, 3)
        sRGB data (batch, px, py, bands)
    """
    print('Load sRGB data (true)')
    srgb = np.load(CAVE_DATA_ROOT + 'cave_srgb.npy')
    return srgb


def get_crgb():
    """load cave cRGB
    
    Returns
    -------
    numpy.array(32, 512, 512, 3)
        cRGB data (batch, px, py, bands)
    """    
    print('Load cRGB (canon20D)')
    crgb = np.load(CAVE_DATA_ROOT + 'cave_crgb.npy')
    return crgb


def get_srgb_gain():
    """load cave sRGB gains
    
    Returns
    -------
    numpy.array(32, 1)
        sRGB gain data (batch, 1)
    """    
    print('Load sRGB gain')
    g_t = np.load(CAVE_DATA_ROOT + 'gt_srgb.npy')
    return g_t
    

def get_crgb_gain():
    """load cave cRGB gains
    
    Returns
    -------
    numpy.array(32, 1)
        cRGB gain data (batch, 1)
    """    
    print('Load cRGB gain(canon20D)')
    g_t= np.load(CAVE_DATA_ROOT + 'gt_crgb.npy')
    return g_t


def get_srgb_gain_s():
    """load cave sRGB normalize gain
    
    Returns
    -------
    numpy.array(1)
        sRGB gain data (1)
    """        
    print('Load srgb gain')
    g_s = np.load(CAVE_DATA_ROOT + 'gs_srgb.npy')
    return g_s
    

def get_crgb_gain_s():
    """load cave cRGB normalize gain
    
    Returns
    -------
    numpy.array(1)
        cRGB gain data (1)
    """       
    print('Load crgb gain(canon20D)')
    g_s= np.load(CAVE_DATA_ROOT + 'gs_crgb.npy')
    return g_s


def get_chart_crgb_gain():
    """load cave cRGB gain
    
    Returns
    -------
    numpy.array(1)
        cRGB chart gain data (1)
    """
    print('Load chart cRGB gain')
    g_s= np.load(CAVE_DATA_ROOT + 'chart_gt_crgb.npy')
    return g_s


def get_chart_srgb_gain():
    """load cave sRGB gain
    
    Returns
    -------
    numpy.array(1)
        sRGB chart gain data (1)
    """
    print('Load chart sRGB gain')
    g_s= np.load(CAVE_DATA_ROOT + 'chart_gt_srgb.npy')
    return g_s


def get_val_crgb():
    """load cRGB validation data
    
    Returns
    -------
    pickle
        validation data (GT=cRGB)
    """
    with open(CAVE_DATA_ROOT + 'val_data/crgb/val_data.pickle', mode='rb') as fi:
        val_data = pickle.load(fi)
    return val_data


def get_val_srgb(camera_name, noise):
    """load sRGB validation data
    
    Returns
    -------
    pickle
        validation data (GT=sRGB)
    """
    with open(CAVE_DATA_ROOT + 'val_data/srgb/' + camera_name + '/val_data_noise' + str(noise) + '.pickle', mode='rb') as fi:
        val_data = pickle.load(fi)
    return val_data