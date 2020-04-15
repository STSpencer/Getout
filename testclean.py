import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_wavelet, denoise_bilateral
import h5py
import glob

hexmethod='oversampling'

onlyfiles = sorted(glob.glob('/store/spencers/Data/Crabrun2/*.hdf5'))

hdffile=onlyfiles[0]

inputdata=h5py.File(hdffile,'r')
trainarr = np.asarray(inputdata[hexmethod][:, :, :, :])

for x in np.arange(np.shape(trainarr)[0]):
    chargevals = []
    for y in np.arange(4):
        chargevals.append(np.sum(trainarr[x,y,:,:]))
    chargevals = np.argsort(chargevals)
    chargevals = np.flip(chargevals,axis=0) #Flip to descending order.
    trainarr[x, :, :, :] = trainarr[x, chargevals, :, :]
    for y in np.arange(4):
        im1=trainarr[x,y,:,:]
        im1=im1[:,:,0]
        print(np.min(im1))
        im1=im1+abs(np.min(im1))
        print(np.min(im1))
        im2=denoise_bilateral(im1)
        print(im2)
        plt.subplot(121)
        print(np.shape(im1))
        plt.imshow(im1)
        plt.subplot(122)
        plt.imshow(im2)
        plt.show()
inputdata.close()
