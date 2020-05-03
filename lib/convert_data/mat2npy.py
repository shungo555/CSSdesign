import numpy as np
import scipy.io
import argparse


def mat2npy(mat_file_name, npy_file_name, variable_name):
    """convert mat data to npy data 
    
    Parameters
    ----------
    mat_file_name : string
        mat file name
    npy_file_name : string
        npy file name
    variable_name : string
        variable name
    """
    data0 =scipy.io.loadmat(mat_file_name)
    data = data0[variable_name]
    print(np.amax(data))
    np.save(npy_file_name, data)
   

def main(args):
    print("convert mat data to npy")
    mat2npy(args.mat, args.npy, args.name)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--npy',dest='npy',help='input npy filename')
    parser.add_argument('--mat',dest='mat',help='output mat filename')
    parser.add_argument('--name',dest='name',help='output variable name')
    args = parser.parse_args()
    main(args)