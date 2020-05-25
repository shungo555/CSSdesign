import numpy as np
import scipy.io
import csv


DATA_ROOT = 'D:/workspace/大学/奥富田中研究室/program/dataset/'
CAMERA_DATA_ROOT = DATA_ROOT + '/camera_data/response/'


def get_camera_sensitivity(camera_name="Canon20D", wavelength_range=[400, 700]):
    """Load canon20D sensitivity
    
    Parameters
    ----------
    camera_name : string, optional
        camera spectral sensitivity, by default Canon20D
    wavelength_range : list, optional
        wave range, by default [400, 700]
    
    Returns
    -------
    numpy.array(31, 3)
        camera sensitivity
    """
    print('Load Camera sensitivity(' + camera_name + ')')  
    data_range = [int((wavelength_range[0] - 400) / 10), 1 + int((wavelength_range[1] - 400) / 10)] 
    sens = np.load(CAMERA_DATA_ROOT + camera_name + '.npy')[:, data_range[0]:data_range[1]]
    return sens.T

if __name__ == "__main__":
    get_camera_sensitivity(camera_name="Canon20D", wavelength_range=[400, 700])
    get_camera_sensitivity(camera_name="NikonD40", wavelength_range=[400, 700])