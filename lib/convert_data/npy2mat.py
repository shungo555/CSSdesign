import numpy as np
import scipy.io
import argparse


def npy2mat(npy_file_name, mat_file_name, variable_name):
    """convert npy data to mat data 
    
    Parameters
    ----------
    npy_file_name : string
        npy file name
    mat_file_name : string
        mat file name
    variable_name : string
        variable name
    """
    npy = np.load(npy_file_name)
    print(np.amax(npy))
    scipy.io.savemat(mat_file_name, {variable_name:npy})
   

def main(args):
    print("convert npy data to mat")
    npy2mat(args.npy,args.mat,args.name)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--npy',dest='npy',help='input npy filename')
    parser.add_argument('--mat',dest='mat',help='output mat filename')
    parser.add_argument('--name',dest='name',help='output variable name')
    args = parser.parse_args()
    main(args)