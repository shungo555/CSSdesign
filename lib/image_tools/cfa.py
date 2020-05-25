import numpy as np


def cfa_bayer(psize, offset = [0, 0], wide_color='g'):
    """ get CFA pattern (bayer)
    
    Parameters
    ----------
    psize :  list 
        [px, py]
    offset : list, optional
        [offset_x, offset_y], by default [0, 0]
    wide_color : string, optional
        bayer wide color, by default 'g'
    Returns
    -------
    CF : array(1, px, py, filter_bands)
        mask of bayer (0 or 1)
    """
    px_s = psize[0]
    py_s = psize[1]       
    RF = np.zeros((px_s, py_s))
    GF = np.zeros((px_s, py_s))
    BF = np.zeros((px_s, py_s))
    px = False
    py = False
    offset_x = offset[0]
    offset_y = offset[1]
    if offset_x % 2 == 1:
        px = True
    if offset_y % 2 == 1:
        py = True
            
    for x in range(px_s):
        for y in range(py_s):

            if wide_color=='r':
                if (x % 2 != px and y % 2 != py):
                    GF[x, y] = 1
                if (x % 2 == px and y % 2 == py):
                    BF[x, y] = 1
                if ((x + y) % 2 != (px ^ py)):
                    RF[x, y] = 1
            elif wide_color=='b':
                if (x % 2 != px and y % 2 != py):
                    GF[x, y] = 1
                if (x % 2 == px and y % 2 == py):
                    RF[x, y] = 1
                if ((x + y) % 2 != (px ^ py)):
                    BF[x, y] = 1
            else:
                if (x % 2 != px and y % 2 != py):
                    RF[x, y] = 1
                if (x % 2 == px and y % 2 == py):
                    BF[x, y] = 1
                if ((x + y) % 2 != (px ^ py)):
                    GF[x, y] = 1

    CF = np.zeros((1, px_s, py_s, 3))
    CF[0, :, :, 0] = RF
    CF[0, :, :, 1] = GF
    CF[0, :, :, 2] = BF
    
    return CF

if __name__ == "__main__":
    f = cfa_bayer([4,4], offset = [0, 0], wide_color='r')
    print('r')
    print(f[0,:,:,0])
    print('g')
    print(f[0,:,:,1])
    print('b')
    print(f[0,:,:,2])