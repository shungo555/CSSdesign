import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse


# DATA_ROOT = '../../'


def smoothness(weight):
    """calculate smoothness term 

    Arguments:
        weight numpy.array(31, 3) -- [description]
    """
    C = np.zeros((weight.shape[0], weight.shape[0]))
    for i in range(weight.shape[0]):
        if not (i == 0 or i == weight.shape[0] - 1):
            C[i, i] = -2
            C[i, i - 1] = 1    
            C[i, i + 1] = 1

    Cw = np.dot(C, weight)
    sm = np.dot(Cw[:,0], Cw[:,0]) + np.dot(Cw[:,1], Cw[:,1]) + np.dot(Cw[:,2], Cw[:,2])  
    return sm


def plot_cpsnr_log(cpsnr_log, val_cpsnr_log, title):
    """plot cpsnr log
    
    Parameters
    ----------
    cpsnr_log : list
        training cpsnr data
    val_cpsnr_log : list
        validation cpsnr data
    title : string
        file name
    """
    plt.figure(num=1, clear=True)
    plt.title('mean_cpsnr')
    plt.xlabel('epoch')
    plt.ylabel('cpsnr')
    plt.plot(cpsnr_log, label='cpsnr')
    plt.plot(val_cpsnr_log, label='val_cpsnr')
    plt.legend()
    plt.savefig(title)


def plot_rmse_log(rmse_log, val_rmse_log, title):
    """plot rmse log
    
    Parameters
    ----------
    rmse_log : list
        training rmse data
    val_rmse_log : list
        validation rmse data
    title : string
        file name
    """
    plt.figure(num=1, clear=True)
    plt.title('mean_rmse')
    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.plot(rmse_log, label='rmse')
    plt.plot(val_rmse_log, label='val_rmse')
    plt.legend()
    plt.savefig(title)


def plot_sensitivity(sensitivity, wavelength_range, title):
    """plot Camera spectral sensitivity
    
    Parameters
    ----------
    sensitivity : numpy.array(31, 3)
        sensitivity data
    wavelength_range : list
        wavelength range
    title : string
        file name
    """
    lambda_band = np.linspace(wavelength_range[0], wavelength_range[1], num = 31)
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111)
    ax.plot(lambda_band, sensitivity[:,0], color='r')
    ax.plot(lambda_band, sensitivity[:,1], color='g')
    ax.plot(lambda_band, sensitivity[:,2], color='b')
    plt.title("Sensitivity")
    plt.xlabel("wavelength [nm]")
    plt.ylabel("sensitivity")
    plt.savefig(title)


def load_csv_data(file_name, column):
    """load csv data

    Parameters
    ----------
    file_name : string
        csv file name
    column : int

    Returns
    -------
    numpy.array()
        log data (column)
    """
    csv_file = open(file_name,'r')
    a_list = [] 
    for row in csv.reader(csv_file):
        a_list.append(row[column]) 
    del a_list[0]
    data = np.zeros((len(a_list),1))
    for i in range(len(a_list)):
        data[i, 0] = a_list[i]
    return data


def plot_all_log(file_name):
    """plot rmse and cpsnr (log data)

    Parameters
    ----------
    file_name : string
        csv file name
    """
    rmse_log = load_csv_data(file_name, 3)
    cpsnr_log = load_csv_data(file_name, 4)  
    val_rmse_log = load_csv_data(file_name, 5)  
    val_cpsnr_log = load_csv_data(file_name, 6)

    plot_rmse_log(rmse_log, val_rmse_log, 'rmse.png')
    plot_cpsnr_log(cpsnr_log, val_cpsnr_log, 'cpsnr.png')


def main(args):
    plot_all_log(args.input, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input', dest='input', default='data.csv', help='input data')
    parser.add_argument('--output', dest='output', default='results/', help='output dir')
    args = parser.parse_args()
    main(args)    