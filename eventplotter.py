import matplotlib.pyplot as plt
import numpy as np
import h5py

datafile=h5py.File('/store/spencers/Data/smalltest/Crabrun2_191_SIM.hdf5','r')
plotmode='oversampling'

isGam=datafile['isGamma'][:].tolist()
ids=datafile['id'][:].tolist()
images=datafile[plotmode][:]
print(np.shape(images))
datafile.close()


for i in np.arange(len(isGam)):
    fig,axs=plt.subplots(2,2)
    plt.suptitle('Event: '+str(ids[i])+' IsGamma: '+str(isGam[i]))
    axs[0,0].imshow(images[i,0,:,:,0])
    axs[0,1].imshow(images[i,1,:,:,0])
    axs[1,0].imshow(images[i,2,:,:,0])
    axs[1,1].imshow(images[i,3,:,:,0])
    plt.show()
