import numpy as np
import scipy.io
import csv
import matplotlib.pyplot as plt

DATA_ROOT = 'D:/workspace/大学/奥富田中研究室/program/dataset/'
ILLUMINANT_DATA_ROOT = DATA_ROOT + '/illuminant_data/'

def get_illuminant(illuminant_name="D65", wavelength_range=[400, 700]):
    """Load illuminant
        
    Parameters
    ----------
    wavelength_range : list, optional
        wave range, by default [400, 700]
    
    Returns
    -------
    numpy.array(31, 31)
        illuminant data (diagonal matrix)
    """   
    print('Load illuminant(' + illuminant_name + ')')  
    data_range = [int((wavelength_range[0] - 400) / 10), 1 + int((wavelength_range[1] - 400) / 10)] 
    L = np.load(ILLUMINANT_DATA_ROOT + illuminant_name + '.npy')[data_range[0]:data_range[1]]
    L = L / np.amax(L)
    L = L.flatten()
    return np.diag(L)


def main():
    wavelength_range=[400, 700]
    data = get_illuminant(illuminant_name="A", wavelength_range=[400, 700])
    lam = np.arange(wavelength_range[0], wavelength_range[1] + 10, 10)
    plt.plot(lam, np.diag(data))
    plt.show()

if __name__ == "__main__":
    main()