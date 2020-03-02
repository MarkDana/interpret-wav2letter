tkn_dict={0: '|', 1: "'", 2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 9: 'h', 10: 'i', 11: 'j', 12: 'k', 13: 'l', 14: 'm', 15: 'n', 16: 'o', 17: 'p', 18: 'q', 19: 'r', 20: 's', 21: 't', 22: 'u', 23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z', 28: '.', 29: '.', 30: '.'}
#此时pre0和out0 都是26dim的

import numpy as np
import os

def get_tkn(filedir):
    pre=os.path.join(filedir,'preOutput.txt')
    pre0=os.path.join(filedir,'preOutput_zero.txt')

    out=os.path.join(filedir,'zeroOutput_999.txt')
    out0=os.path.join(filedir,'zeroOutput_999.txt')

    writedir=os.path.join(filedir,'write_token.txt')

    tkn_mtx=np.loadtxt(pre,comments=['p','[','#','l'])
    tkn_mtx=tkn_mtx.T
    with open(writedir,'a')as f:
        f.write('pre:\n')
        for frame in tkn_mtx:
            ind=np.argmax(frame)
            f.write(tkn_dict[ind])
        f.write('\n\n')

    tkn_mtx = np.loadtxt(pre0, comments=['p', '[', '#', 'l'])
    tkn_mtx = tkn_mtx.T
    with open(writedir, 'a')as f:
        f.write('pre0:\n')
        for frame in tkn_mtx:
            ind = np.argmax(frame)
            f.write(tkn_dict[ind+2])
        f.write('\n\n')

    tkn_mtx = np.loadtxt(out, comments=['p', '[', '#', 'l'])
    tkn_mtx = tkn_mtx.T
    with open(writedir, 'a')as f:
        f.write('out:\n')
        for frame in tkn_mtx:
            ind = np.argmax(frame)
            f.write(tkn_dict[ind])
        f.write('\n\n')

    tkn_mtx = np.loadtxt(out0, comments=['p', '[', '#', 'l'])
    tkn_mtx = tkn_mtx.T
    with open(writedir, 'a')as f:
        f.write('out0:\n')
        for frame in tkn_mtx:
            ind = np.argmax(frame)
            f.write(tkn_dict[ind+2])
        f.write('\n\n')


if __name__ == '__main__':
    wholeDir='/Users/markdana/Desktop/data'

    for folder in os.listdir(wholeDir):
        if folder in ['.DS_Store','000000000.flac','000000000.tkn']:continue
        get_tkn(os.path.join(wholeDir,folder))

