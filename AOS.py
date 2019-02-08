import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal



class AOS:
    def __init__(self, alpha = 25, outer_iter = 25):
        self.alpha = alpha
        self.outer_iter = outer_iter
    
    @classmethod
    def imfilter(self,im, f):

        imrows, imcols, imchannels = im.shape;
        retval = np.zeros_like(im, dtype="float")
        frows, fcols = f.shape
        padding = np.floor([frows/2, fcols/2])
        im = np.pad(im,((int(padding[0]), int(padding[0])), (int(padding[1]), int(padding[1])), (0,0)),'edge')

        if frows % 2 == 0:
            im = im[1:-1, :, :]
        if fcols % 2 == 0:
            im = im[:, 1:-1, :] 

        for i in range(2,-1,-1):
            retval[:,:,i] = signal.convolve2d(im[:,:,i], f, 'valid')
        return retval

    @classmethod   
    def TDMA(self, a,b,c,d):

        rows, cols = a.shape
        c[0,:] = c[0,:] / b[0, :]
        d[0,:] = d[0, :] / b[0, :]

        for i in range(1, rows):
            temp = 1 / (b[i,:] - a[i,:] * c[i-1,:])
            c[i,:] = c[i,:] * temp
            d[i,:] = (d[i,:] - a[i,:] * d[i-1,:]) * temp

        d[rows-1,:] = ( d[rows-1,:] - a[rows-1,:] * d[rows-2,:] )/( b[rows-1,:] - a[rows-1,:] * c[rows-2,:] )
        x = np.zeros_like(d)

        x[rows-1,:] = d[rows-1,:]
        for i in range(rows-2,-1,-1):
            x[i,:] = d[i,:] - c[i,:] * x[i + 1,:]

        return x
    @classmethod
    def DiffWeights(self, img):
        rows, cols, frames = img.shape
        Dver = self.imfilter(img, np.array([-0.25,0,0.25]).reshape((-1, 1)))
        Dhor = self.imfilter(img, np.array([-0.25,0,0.25]).reshape((1,-1)))

        wW = (np.roll(img, [0,1,0], [0,1, 2])-img)**2 + (Dver + np.roll(Dver, [0,1,0], [0,1, 2]))**2
        wE = (np.roll(img, [0,-1,0], [0,1, 2])-img)**2 + (Dver + np.roll(Dver, [0,-1,0], [0,1, 2]))**2
        wN = (np.roll(img, [1,0,0], [0,1, 2])-img)**2 + (Dhor + np.roll(Dhor, [1,0,0], [0,1, 2]))**2
        wS = (np.roll(img, [-1,0,0], [0,1, 2])-img)**2 + (Dhor + np.roll(Dhor, [-1,0,0], [0,1, 2]))**2

        wW = np.max(wW,2)
        wE = np.max(wE,2)
        wN = np.max(wN,2)
        wS = np.max(wS,2)

        wW = 1 / np.sqrt( wW + 0.00001 )
        wE = 1 / np.sqrt( wE + 0.00001 )
        wN = 1 / np.sqrt( wN + 0.00001 )
        wS = 1 / np.sqrt( wS + 0.00001 )

        wW[:,0] = 0
        wE[:,-1] = 0
        wN[0, :] = 0
        wS[-1, :] = 0

        return wW, wN, wE, wS
    

    def transform(self, img, diff_speed = False):
        img = img.astype("float")
        rows, cols, dims = img.shape
        self.image = img.copy()
        self.img = img
        for i in range(self.outer_iter+1):
            wW, wN, wE, wS = self.DiffWeights(self.img)
            a_ver = - self.alpha*wN
            b_ver = 2 + self.alpha * (wN + wS)
            c_ver = -self.alpha * wS

            a_hor = -self.alpha * wW.T
            b_hor = 2 + self.alpha * (wW + wE).T
            c_hor = -self.alpha * wE.T

            for k in range(dims):
                ver = self.TDMA(a_ver.copy(), b_ver.copy(), c_ver.copy(), img[:,:,k].copy())
                hor = self.TDMA(a_hor.copy(), b_hor.copy(), c_hor.copy(), img[:,:,k].copy().T).T
                img[:,:,k] = ver + hor
        
        if diff_speed:
            return np.abs((np.log(self.image.astype('float')+1))-(np.log(img.astype("float")+1)))
        else:                 
            return img.astype("uint8")
                
