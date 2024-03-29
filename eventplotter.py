import matplotlib.pyplot as plt
import numpy as np
import h5py

datafile=h5py.File('/store/spencers/Data/Crab64080/Crabrun2_132_SIM.hdf5','r')
plotmode='oversampling'
print(datafile.keys)
isGam=datafile['isGamma'][:].tolist()
ids=datafile['id'][:].tolist()

images=datafile[plotmode][:]
print(np.shape(images))
datafile.close()
print(ids)
for i in np.arange(len(isGam)):
    fig,axs=plt.subplots(2,2)
    if isGam[i]==1:
        plt.suptitle('Simulated $\gamma$-Ray, Event Number: '+str(ids[i]))
    else:
        plt.suptitle('Simulated Proton, Event Number: '+str(ids[i]))
    #plt.suptitle('Real Event, Event Number: '+str(ids[i]))
    im1=axs[0,0].imshow(images[i,0,:,:,0])
    plt.colorbar(im1,ax=axs[0,0],label='Intensity (p.e.)')
    axs[0,0].set_xlabel('x (Pixels)')
    axs[0,0].set_ylabel('y (Pixels)')
    im2=axs[0,1].imshow(images[i,1,:,:,0])
    plt.colorbar(im2,ax=axs[0,1],label='Intensity (p.e.)')
    axs[0,1].set_xlabel('x (Pixels)')
    axs[0,1].set_ylabel('y (Pixels)')
    im3=axs[1,0].imshow(images[i,2,:,:,0])
    axs[1,0].set_xlabel('x (Pixels)')
    axs[1,0].set_ylabel('y (Pixels)')
    plt.colorbar(im3,ax=axs[1,0],label='Intensity (p.e.)')
    im4=axs[1,1].imshow(images[i,3,:,:,0])
    axs[1,1].set_xlabel('x (Pixels)')
    axs[1,1].set_ylabel('y (Pixels)')
    plt.colorbar(im4,ax=axs[1,1],label='Intensity (p.e.)')
    plt.tight_layout()
    plt.savefig('/store/spencers/eventplots/sim_'+str(i)+'_oversampling.png')
    plt.clf()
    plt.cla()
