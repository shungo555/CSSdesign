import numpy as np
import scipy.io
import csv


DATA_ROOT = 'D:/workspace/大学/奥富田中研究室/program/dataset/'
CAMERA_DATA_ROOT = DATA_ROOT + 'camera_data/'

def get_illuminant(wavelength_range=[400, 700]):
    """Load illuminant (D65)
        
    Parameters
    ----------
    wavelength_range : list, optional
        wave range, by default [400, 700]
    
    Returns
    -------
    numpy.array(31, 31)
        illuminant data (diagonal matrix)
    """   
    print('Load illuminant (D65)')  
    data_range = [int((wavelength_range[0] - 400) / 10), 1 + int((wavelength_range[1] - 400) / 10)] 
    L = np.load(CAMERA_DATA_ROOT + 'D65.npy')[data_range[0]:data_range[1], 0]
    L = L / np.amax(L)
    L = L.flatten()
    return np.diag(L)


def get_xyz(wavelength_range=[400, 700]):
    """Load xyz function
    
    Parameters
    ----------
    wavelength_range : list, optional
        wave range, by default [400, 700]
    
    Returns
    -------
    numpy.array(3, 31)
        xyz data
    """
    print('Load xyz function')    
    data_range = [int((wavelength_range[0] - 400) / 10), 1 + int((wavelength_range[1] - 400) / 10)]
    bar = scipy.io.loadmat(CAMERA_DATA_ROOT  + 'xyzbar.mat')
    xyzbar = bar['xyzbar']
    xyzbar = xyzbar[data_range[0]:data_range[1],:]
    xyz = xyzbar / np.amax(xyzbar)
    xyz = np.transpose(xyz, (1, 0))
    return xyz


def get_invM():
    """Load convert matrix from xyz to sRGB
    
    Returns
    -------
    numpy.array(3, 3)
        convert matrix data
    """
    print('Load invM')
    invM_data = scipy.io.loadmat(CAMERA_DATA_ROOT  + 'invM.mat')
    invM = invM_data['invM']
    return invM

        
def get_chart_hsi():
    """Load chart HSI
    
    Returns
    -------
    numpy.array(128, 192, 31)
        chart HSI data
    """
    print('Load chart HSI')
    chart = np.load(CAMERA_DATA_ROOT + 'chart_reflectance.npy')
    return chart


def get_chart():
    """Load chart reflectance data
    
    Returns
    -------
    numpy.array(31, 96)
        chart reflectance data
    """    
    print('Load chart')
    with open(CAMERA_DATA_ROOT + 'chart.csv') as f:
        reader = csv.reader(f)
        R0 = np.array([row for row in reader], dtype='float64')
    R =  R0[2:33,:]
    return R
    

def get_H(max_gain=0.95, wavelength_range=[400, 700]):
    """Load sRGB sensitivity H
    
    Returns
    -------
    numpy.array(3, 31)
        sRGB sensitivity
    """
    print('Load sRGB sensitivity H')
    xyz = get_xyz(wavelength_range)
    invM = get_invM()
    H = np.dot(invM, xyz)
    return H


if __name__ == "__main__":
    pass